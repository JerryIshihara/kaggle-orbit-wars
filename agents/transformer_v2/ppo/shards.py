"""Rollout-shard serialization + GCS transport + trainable-delta handoff.

This is the shared data layer for the distributed PPO protocol (Colab learner +
GCP CPU actor + GCS transport; see docs/PPO_TWO_CPU_PROTOCOL.md and the plan in
~/.claude/plans). It has no heavy imports at module load: the ``smoke``/``sampler``
dataclasses are imported lazily inside ``dict_to_episode_buffer`` so this module is
usable on macOS (where the Linux-only ``kaggle_environments`` env cannot run) for
the serialization round-trip + the trainable-delta + GCS helpers.

Three responsibilities:
  1. EpisodeBuffer <-> dict-of-CPU-tensors  (``save_shard`` / ``load_shard``)
     with a hard version check (policy_version / action_contract / history_window).
  2. The <1 MB "trainable-only" policy delta the VM downloads each iteration
     (``save_trainable_delta`` / ``load_trainable_delta``) — exactly the
     ``requires_grad`` params that ``learner.build_optimizer`` optimizes.
  3. Thin GCS helpers (shell out to ``gcloud storage`` when present, else fall
     back to the ``google-cloud-storage`` library on hosts without the gcloud CLI)
     + a tiny ``progress.json`` publish/read pair so the VM's live rollout
     progress is visible from Colab.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

# Bumped if the on-disk shard schema changes incompatibly.
SHARD_FORMAT_VERSION = 3
# Matches the sampler's action contract (sampler.MultiTargetAction /
# sample_multi_target).
POLICY_ACTION_CONTRACT = "bernoulli_select_multinomial_alloc_v2"

# Per-step StepRecord fields (smoke.StepRecord). Tensor fields are stacked along a
# new leading step axis; for T=1 each is (P,...)/(F,...), for T=10 (T,P,...) etc.
STEP_TENSOR_FIELDS = (
    "planet_features", "fleet_features",
    "fleet_target_idx", "fleet_source_idx", "fleet_owner_slot",
    "fleet_ships_log", "fleet_eta_norm", "fleet_mask",
    "planet_mask", "is_comet", "pair_type_ids",
    "pair_mask", "source_mask",
)
STEP_SCALAR_FIELDS = (
    "value", "reward", "done",
    "invalid_launch", "emitted_launch", "n_selected_targets",
    "score_my", "score_enemy_max", "phi",
)
# Action fields (sampler.MultiTargetAction). bernoulli_select_multinomial_alloc_v2:
#   select_mask (P, P) bool — fired off-diagonal targets per owned source,
#   alloc_counts (P, P) long — ships routed to each fired target,
#   self_counts (P,) long — ships held on each owned source (self category).
ACTION_TENSOR_FIELDS = ("select_mask", "alloc_counts", "self_counts")
# bounded_k_select_multinomial_alloc_v3 (sampler.BoundedKAction):
#   select_counts (P, P+1) long — draws per [targets..., self token],
#   alloc_extras (P, P) long — extras beyond the min_launch floor per fired,
#   self_extras (P,) long — extras held on acting rows.
ACTION_TENSOR_FIELDS_V3 = ("select_counts", "alloc_extras", "self_extras")
# bounded_k_select_dirichlet_alloc_v4 (sampler.DirichletKAction):
#   select_counts (P, P+1) long — as v3,
#   alloc_shares (P, P+1) float32 — the stored ε-clamped Dirichlet draw
#   (float32 REQUIRED: the logprob recompute evaluates log x at these values,
#   so any dtype round-trip would desync the PPO ratio).
ACTION_TENSOR_FIELDS_V4 = ("select_counts", "alloc_shares",
                           "select_row_logp", "alloc_row_logp")
ACTION_LOGPROB_FIELDS = ("logprob", "logprob_select", "logprob_alloc")


class ShardVersionError(RuntimeError):
    """Raised when a shard / delta fails a hard version or contract check."""


# --------------------------------------------------------------------------- #
# git sha + sha256 helpers
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    """Current repo HEAD sha, or 'unknown' (the VM runs from code.tgz, not a checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# EpisodeBuffer <-> dict
# --------------------------------------------------------------------------- #
def episode_buffer_to_dict(ep: Any) -> dict[str, Any]:
    """Stack one EpisodeBuffer's steps into batched CPU tensors (duck-typed; no import)."""
    steps = list(ep.steps)
    n = len(steps)
    d: dict[str, Any] = {
        "seed": int(ep.seed),
        "learner_seat": int(ep.learner_seat),
        "winner": (None if ep.winner is None else int(ep.winner)),
        "n_steps": n,
    }
    if n == 0:
        return d
    for f in STEP_TENSOR_FIELDS:
        d[f] = torch.stack([getattr(s, f).detach().cpu() for s in steps], dim=0)
    for f in STEP_SCALAR_FIELDS:
        d[f] = torch.tensor([getattr(s, f) for s in steps])
    d["invalid_reasons"] = [list(s.invalid_reasons or []) for s in steps]
    is_v4 = hasattr(steps[0].action, "alloc_shares")
    is_v3 = (not is_v4) and hasattr(steps[0].action, "select_counts")
    d["action_contract"] = "v4" if is_v4 else ("v3" if is_v3 else "v2")
    fields = (ACTION_TENSOR_FIELDS_V4 if is_v4
              else ACTION_TENSOR_FIELDS_V3 if is_v3 else ACTION_TENSOR_FIELDS)
    for f in fields:
        d["action_" + f] = torch.stack(
            [getattr(s.action, f).detach().cpu() for s in steps], dim=0
        )
    for f in ACTION_LOGPROB_FIELDS:
        d["action_" + f] = torch.stack(
            [getattr(s.action, f).detach().cpu().reshape(()) for s in steps], dim=0
        )
    d["action_n_terms"] = torch.tensor(
        [int(s.action.n_terms) for s in steps], dtype=torch.long
    )
    d["action_diagnostics"] = [dict(s.action.diagnostics or {}) for s in steps]
    return d


def dict_to_episode_buffer(d: dict[str, Any]) -> Any:
    """Rebuild an EpisodeBuffer (list[StepRecord]) from ``episode_buffer_to_dict``."""
    from .sampler import BoundedKAction, DirichletKAction, MultiTargetAction  # lazy import
    from .smoke import EpisodeBuffer, StepRecord

    n = int(d["n_steps"])
    is_v4 = d.get("action_contract") == "v4" or "action_alloc_shares" in d
    is_v3 = (not is_v4) and (
        d.get("action_contract") == "v3" or "action_select_counts" in d)
    steps: list[Any] = []
    for i in range(n):
        if is_v4:
            action = DirichletKAction(
                select_counts=d["action_select_counts"][i].long(),
                alloc_shares=d["action_alloc_shares"][i].float(),
                select_row_logp=d["action_select_row_logp"][i].float()
                if "action_select_row_logp" in d else None,
                alloc_row_logp=d["action_alloc_row_logp"][i].float()
                if "action_alloc_row_logp" in d else None,
                logprob=d["action_logprob"][i],
                logprob_select=d["action_logprob_select"][i],
                logprob_alloc=d["action_logprob_alloc"][i],
                n_terms=int(d["action_n_terms"][i].item()),
                diagnostics=dict(d["action_diagnostics"][i]),
            )
        elif is_v3:
            action = BoundedKAction(
                select_counts=d["action_select_counts"][i].long(),
                alloc_extras=d["action_alloc_extras"][i].long(),
                self_extras=d["action_self_extras"][i].long(),
                logprob=d["action_logprob"][i],
                logprob_select=d["action_logprob_select"][i],
                logprob_alloc=d["action_logprob_alloc"][i],
                n_terms=int(d["action_n_terms"][i].item()),
                diagnostics=dict(d["action_diagnostics"][i]),
            )
        else:
            action = MultiTargetAction(
                select_mask=d["action_select_mask"][i].bool(),
                alloc_counts=d["action_alloc_counts"][i].long(),
                self_counts=d["action_self_counts"][i].long(),
                logprob=d["action_logprob"][i],
                logprob_select=d["action_logprob_select"][i],
                logprob_alloc=d["action_logprob_alloc"][i],
                n_terms=int(d["action_n_terms"][i].item()),
                diagnostics=dict(d["action_diagnostics"][i]),
            )
        kw = {f: d[f][i] for f in STEP_TENSOR_FIELDS}
        steps.append(StepRecord(
            action=action,
            value=float(d["value"][i].item()),
            reward=float(d["reward"][i].item()),
            done=float(d["done"][i].item()),
            invalid_launch=int(d["invalid_launch"][i].item()),
            emitted_launch=int(d["emitted_launch"][i].item()),
            n_selected_targets=int(d["n_selected_targets"][i].item()),
            invalid_reasons=list(d["invalid_reasons"][i]),
            score_my=int(d["score_my"][i].item()),
            score_enemy_max=int(d["score_enemy_max"][i].item()),
            phi=float(d.get("phi", torch.zeros(n))[i].item()) if n else 0.0,
            **kw,
        ))
    return EpisodeBuffer(
        seed=int(d["seed"]),
        learner_seat=int(d["learner_seat"]),
        steps=steps,
        winner=(None if d["winner"] is None else int(d["winner"])),
    )


# --------------------------------------------------------------------------- #
# Shard save / load + manifest
# --------------------------------------------------------------------------- #
def save_shard(
    ep: Any,
    path: str | Path,
    *,
    policy_version: int,
    history_window: int,
    git_sha_str: str | None = None,
    vm_id: str = "",
    opponent_ckpt: str = "",
) -> dict[str, Any]:
    """Serialize one EpisodeBuffer to ``path``; return its manifest entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep_dict = episode_buffer_to_dict(ep)
    # Stamp the contract the episode was actually sampled under (the dict
    # carries "v3" for BoundedKAction steps); v2 keeps the module constant.
    contract = {
        "v4": "bounded_k_select_dirichlet_alloc_v4",
        "v3": "bounded_k_select_multinomial_alloc_v3",
    }.get(ep_dict.get("action_contract"), POLICY_ACTION_CONTRACT)
    payload = {
        "format_version": SHARD_FORMAT_VERSION,
        "policy_version": int(policy_version),
        "policy_action_contract": contract,
        "history_window": int(history_window),
        "git_sha": git_sha_str or git_sha(),
        "vm_id": vm_id,
        "opponent_ckpt": opponent_ckpt,
        "episode": ep_dict,
    }
    torch.save(payload, path)
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "seed": int(ep.seed),
        "seat": int(ep.learner_seat),
        "winner": (None if ep.winner is None else int(ep.winner)),
        "n_steps": len(ep.steps),
    }


def load_shard(
    path: str | Path,
    *,
    expect_policy_version: int | None = None,
    expect_contract: str | None = None,
    expect_history_window: int | None = None,
) -> Any:
    """Load + version-check a shard; return its EpisodeBuffer. Hard-fail on skew."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pv = payload.get("policy_version")
    ct = payload.get("policy_action_contract")
    hw = payload.get("history_window")
    if expect_policy_version is not None and pv != expect_policy_version:
        raise ShardVersionError(
            f"shard policy_version {pv} != expected {expect_policy_version}: {path}"
        )
    if expect_contract is not None and ct != expect_contract:
        raise ShardVersionError(
            f"shard action_contract {ct!r} != expected {expect_contract!r}: {path}"
        )
    if expect_history_window is not None and hw != expect_history_window:
        raise ShardVersionError(
            f"shard history_window {hw} != expected {expect_history_window}: {path}"
        )
    return dict_to_episode_buffer(payload["episode"])


def write_manifest(
    path: str | Path,
    entries: list[dict[str, Any]],
    *,
    policy_version: int,
    history_window: int,
    git_sha_str: str | None = None,
    vm_id: str = "",
) -> dict[str, Any]:
    manifest = {
        "policy_version": int(policy_version),
        "policy_action_contract": POLICY_ACTION_CONTRACT,
        "history_window": int(history_window),
        "git_sha": git_sha_str or git_sha(),
        "vm_id": vm_id,
        "shards": list(entries),
    }
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Trainable-only delta (the <1 MB head update the VM downloads per iter)
# --------------------------------------------------------------------------- #
def save_trainable_delta(
    policy: torch.nn.Module,
    path: str | Path,
    *,
    base_version: str,
    git_sha_str: str | None = None,
) -> dict[str, Any]:
    """Save only the ``requires_grad`` params (== build_optimizer's set).

    ``base_version`` identifies the frozen backbone these heads sit on; the VM
    refuses to apply a delta whose base does not match its resident base.
    """
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    sd = policy.state_dict()
    delta = {k: v.detach().cpu() for k, v in sd.items() if k in trainable}
    payload = {
        "format": "trainable_delta_v1",
        "base_version": base_version,
        "git_sha": git_sha_str or git_sha(),
        "keys": sorted(delta.keys()),
        "state": delta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "n_keys": len(delta),
        "n_params": int(sum(v.numel() for v in delta.values())),
        "bytes": path.stat().st_size,
    }


def load_trainable_delta(
    model: torch.nn.Module,
    path: str | Path,
    *,
    expect_base_version: str | None = None,
) -> dict[str, Any]:
    """Apply a head-delta onto a resident frozen base. Hard-fail on backbone mismatch."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    base = payload.get("base_version")
    if expect_base_version is not None and base != expect_base_version:
        raise ShardVersionError(
            f"delta base_version {base!r} != resident base {expect_base_version!r}: {path}"
        )
    res = model.load_state_dict(payload["state"], strict=False)
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected:
        raise ShardVersionError(
            f"trainable delta has {len(unexpected)} keys absent from model "
            f"(e.g. {unexpected[:3]}): {path}"
        )
    return {
        "loaded_keys": len(payload["state"]),
        "missing_frozen": len(getattr(res, "missing_keys", [])),
        "base_version": base,
    }


# --------------------------------------------------------------------------- #
# GCS helpers (shell out to gcloud storage when present; otherwise fall back to
# the google-cloud-storage library for hosts without the gcloud CLI, e.g.
# Machine B, which has the lib + application-default credentials). gcloud stays
# the DEFAULT whenever it is on PATH so A/Colab behavior is unchanged.
# --------------------------------------------------------------------------- #
_HAS_GCLOUD = shutil.which("gcloud") is not None

# Lazily-constructed google.cloud.storage.Client cache (only the lib path uses it).
_GCS_CLIENT = None


def _gcloud(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gcloud", "storage", *args],
        check=True, capture_output=True, text=True,
    )


def _gcs_client():
    """Lazy, cached google-cloud-storage Client (used only on gcloud-less hosts)."""
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        import os

        from google.cloud import storage  # lazy: keeps module import light on A/Colab

        project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("PPO_FIRESTORE_PROJECT")
            or None
        )
        _GCS_CLIENT = storage.Client(project=project)
    return _GCS_CLIENT


def _parse_gs(url: str) -> tuple[str, str]:
    """Split ``gs://bucket/name`` -> ``(bucket, name)``."""
    rest = str(url)[len("gs://"):]
    bucket, _, name = rest.partition("/")
    return bucket, name


def _is_gs(p) -> bool:
    return str(p).startswith("gs://")


def gcs_exists(url: str) -> bool:
    if not _is_gs(url):
        return Path(url).exists()
    if not _HAS_GCLOUD:
        client = _gcs_client()
        bucket, name = _parse_gs(url)
        return client.bucket(bucket).blob(name).exists(client)
    try:
        _gcloud(["objects", "describe", url, "--format=value(size)"])
        return True
    except subprocess.CalledProcessError:
        return False


def gcs_size(url: str) -> int | None:
    try:
        out = _gcloud(["objects", "describe", url, "--format=value(size)"])
        return int(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def gcs_cp(src: str | Path, dst: str | Path) -> None:
    s, d = str(src), str(dst)
    if not _is_gs(s) and not _is_gs(d):
        # local -> local: no gcloud needed (two-Mac rsync flow / gcloud-less hosts)
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        return
    if _HAS_GCLOUD:
        _gcloud(["cp", s, d])
        return
    # Library path (gcloud-less host): one of the two ends is gs://.
    client = _gcs_client()
    if _is_gs(s) and not _is_gs(d):
        # gs -> local
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        bucket, name = _parse_gs(s)
        client.bucket(bucket).blob(name).download_to_filename(d)
    elif not _is_gs(s) and _is_gs(d):
        # local -> gs
        bucket, name = _parse_gs(d)
        client.bucket(bucket).blob(name).upload_from_filename(s)
    else:
        # gs -> gs
        sb, sn = _parse_gs(s)
        db, dn = _parse_gs(d)
        src_bucket = client.bucket(sb)
        src_bucket.copy_blob(src_bucket.blob(sn), client.bucket(db), dn)


def gcs_cp_parallel(pairs: list[tuple[str, str]], *, workers: int = 8) -> None:
    """Copy many (src, dst) pairs concurrently. Raises on the first failure."""
    if not pairs:
        return
    n = max(1, min(int(workers), len(pairs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(gcs_cp, s, d) for s, d in pairs]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()


def gcs_upload_dir(local_dir: str | Path, gs_prefix: str) -> None:
    if _is_gs(gs_prefix) and not _HAS_GCLOUD:
        # Library path: walk local_dir, upload each file under the prefix.
        client = _gcs_client()
        bucket_name, base = _parse_gs(gs_prefix.rstrip("/") + "/")
        bucket = client.bucket(bucket_name)
        root = Path(local_dir)
        for p in root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                bucket.blob(base + rel).upload_from_filename(str(p))
        return
    _gcloud(["cp", "--recursive", str(Path(local_dir)) + "/.", gs_prefix.rstrip("/") + "/"])


def gcs_download_dir(gs_prefix: str, local_dir: str | Path) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    if not _is_gs(gs_prefix):
        # local tree copy (mirror gcloud's flat "cp --recursive prefix/* dst")
        src = Path(gs_prefix)
        for p in src.rglob("*"):
            if p.is_file():
                dest = Path(local_dir) / p.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        return
    if not _HAS_GCLOUD:
        # Library path: list blobs under the prefix and download each one.
        client = _gcs_client()
        bucket_name, prefix = _parse_gs(gs_prefix.rstrip("/") + "/")
        bucket = client.bucket(bucket_name)
        for blob in client.list_blobs(bucket, prefix=prefix):
            if blob.name.endswith("/"):
                continue  # skip "directory" placeholder blobs
            suffix = blob.name[len(prefix):]
            dest = Path(local_dir) / suffix
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
        return
    _gcloud(["cp", "--recursive", gs_prefix.rstrip("/") + "/*", str(local_dir)])


def gcs_list(gs_prefix: str) -> list[str]:
    if _is_gs(gs_prefix) and not _HAS_GCLOUD:
        # Library path: list blobs under the prefix as gs:// urls.
        try:
            client = _gcs_client()
            bucket_name, prefix = _parse_gs(gs_prefix)
            return [
                f"gs://{bucket_name}/{blob.name}"
                for blob in client.list_blobs(client.bucket(bucket_name), prefix=prefix)
            ]
        except Exception:
            return []
    try:
        out = _gcloud(["ls", gs_prefix])
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except subprocess.CalledProcessError:
        return []


# --------------------------------------------------------------------------- #
# Live progress (tiny JSON to/from GCS so Colab can tail the VM rollout)
# --------------------------------------------------------------------------- #
def publish_progress(url: str, payload: dict[str, Any]) -> None:
    """Write a small progress.json to a GCS url (best-effort; never raises)."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            tmp = fh.name
        gcs_cp(tmp, url)
    except Exception:
        pass
    finally:
        try:
            Path(tmp).unlink()
        except Exception:
            pass


def read_progress(url: str) -> dict[str, Any] | None:
    """Read a progress.json from GCS (or a local path), or None if absent/unreadable."""
    if not _is_gs(url):
        try:
            return json.loads(Path(url).read_text())
        except Exception:
            return None
    if not _HAS_GCLOUD:
        try:
            client = _gcs_client()
            bucket, name = _parse_gs(url)
            return json.loads(client.bucket(bucket).blob(name).download_as_text())
        except Exception:
            return None
    try:
        out = subprocess.run(
            ["gcloud", "storage", "cat", url],
            check=True, capture_output=True, text=True,
        )
        return json.loads(out.stdout)
    except Exception:
        return None


def utcnow_iso() -> str:
    """UTC timestamp (callers pass time around; kept here for a single source)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
