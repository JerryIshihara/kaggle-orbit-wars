"""Pack the transformer_v2 PairHead policy for Kaggle submission.

Mirrors the package tree the runtime needs (transformer_v2 + heuristic.physical_v4
+ agents top-level shim) into a self-contained ``submissions/transformer_v2/``,
copies the 4 ckpts to ``weights/``, writes a ``main.py`` entry point, and
bundles to ``submissions/transformer_v2.tar.gz``.

The default L0 ckpts under ``data/runs/{planet,fleet,comet}/...`` and the
trainable ckpt under ``data/runs/entity/<run>/entity_encoder_best.pt`` are
copied into the bundle; ``main.py`` passes their bundle-local paths to
``TransformerAgent.load(planet_run_dir=..., fleet_run_dir=..., comet_run_dir=...)``
so no host-side ``data/runs/`` tree is required at submit time.

Run:
    python scripts/pack_transformer_v2_for_kaggle.py
    python scripts/pack_transformer_v2_for_kaggle.py --submit
    python scripts/pack_transformer_v2_for_kaggle.py --submit --note "T=6, 5/5 v4"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Default L0 / entity ckpts the runner expects. CLI overridable.
DEFAULT_ENTITY_CKPT = (
    REPO / "data" / "runs" / "entity"
    / "bowEbi_pair5head_d256_lr5e-05_b128_30ep_20260519-141507"
    / "entity_encoder_best.pt"
)
DEFAULT_PLANET_CKPT = (
    REPO / "data" / "runs" / "planet"
    / "specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep"
    / "planet_encoder_best.pt"
)
DEFAULT_FLEET_CKPT = (
    REPO / "data" / "runs" / "fleet"
    / "specialist_fleet_d256_40k_lr1e4_120ep"
    / "fleet_encoder_best.pt"
)
DEFAULT_COMET_CKPT = (
    REPO / "data" / "runs" / "comet"
    / "fullpath_scalar_multitask_d256_40k_lr1e4_120ep"
    / "comet_past_best.pt"
)

# Minimal shims so the bundle's ``agents`` package can be imported without
# pulling in every heuristic sibling (which would chain into kaggle_environments
# at import time; that's available at submit time, but the chain wastes RAM
# and risks unrelated import errors hiding our agent).
AGENTS_INIT = (
    "from .registry import Agent, AgentSpec, list_agent_specs, list_agents, register\n"
    "from . import heuristic  # noqa: F401 — registers physical_v4\n"
    "from . import transformer_v2  # noqa: F401 — registers the learner\n"
    "__all__ = [\n"
    "    'Agent', 'AgentSpec', 'list_agents', 'list_agent_specs', 'register',\n"
    "]\n"
)
HEURISTIC_INIT = (
    "from . import physical_v4  # noqa: F401 — physical_v4 launch helpers\n"
)


def _copy_tree(src: Path, dst: Path, *, skip_pycache: bool = True) -> None:
    """Copy ``src`` to ``dst`` recursively, skipping ``__pycache__``."""
    if dst.exists():
        shutil.rmtree(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"missing source dir: {src}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc") if skip_pycache else None,
    )


def _git_short_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        )
        return bool(r.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_bundle(
    *,
    out_dir: Path,
    entity_ckpt: Path,
    planet_ckpt: Path,
    fleet_ckpt: Path,
    comet_ckpt: Path,
) -> Path:
    """Materialize ``out_dir`` with a self-contained package + weights.

    Returns the path to the produced ``main.py``.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    bundle_agents = out_dir / "agents"
    bundle_agents.mkdir()

    # Top-level shim files (not full copies of agents/__init__.py — that
    # imports heuristic.__init__ which iterates over every sibling).
    (bundle_agents / "__init__.py").write_text(AGENTS_INIT)
    (bundle_agents / "registry.py").write_bytes(
        (REPO / "agents" / "registry.py").read_bytes()
    )
    (bundle_agents / "physics_utils.py").write_bytes(
        (REPO / "agents" / "physics_utils.py").read_bytes()
    )

    # transformer_v2 package — copy verbatim. Same with physical_v4.
    _copy_tree(
        REPO / "agents" / "transformer_v2",
        bundle_agents / "transformer_v2",
    )

    heuristic_dir = bundle_agents / "heuristic"
    heuristic_dir.mkdir()
    (heuristic_dir / "__init__.py").write_text(HEURISTIC_INIT)
    _copy_tree(
        REPO / "agents" / "heuristic" / "physical_v4",
        heuristic_dir / "physical_v4",
    )

    # Weights — flat under weights/, named exactly as ``_load_encoders`` /
    # the entity-pretrain ckpt convention expects.
    weights_dir = out_dir / "weights"
    weights_dir.mkdir()
    for src, dst_name in (
        (entity_ckpt, "entity_encoder_best.pt"),
        (planet_ckpt, "planet_encoder_best.pt"),
        (fleet_ckpt, "fleet_encoder_best.pt"),
        (comet_ckpt, "comet_past_best.pt"),
    ):
        if not src.exists():
            raise FileNotFoundError(f"missing ckpt: {src}")
        shutil.copy2(src, weights_dir / dst_name)

    # NOTE: Kaggle's agent harness loads ``main.py`` via ``exec(compile(...))``
    # with an empty namespace, so ``__file__`` is NOT defined. Locate the
    # bundle root via ``os.getcwd()`` (the harness ``chdir``s to the agent
    # dir before exec) with fallbacks to ``sys.path[-1]`` (the harness also
    # ``sys.path.append``s the agent dir) and ``/kaggle_simulations/agent``.
    #
    # Also: ``kaggle_environments.envs.lux_ai_s3.lux_ai_s3`` does
    # ``sys.path.append(__dir__)`` at env-registration time, which puts a
    # single-file ``agents.py`` module on the top-level import path. Our
    # bundle's ``agents/`` package gets SHADOWED unless we PREPEND our dir
    # (the harness only ``append``s it). Always insert at position 0 and
    # purge any stale ``agents`` entries from ``sys.modules`` before
    # importing the runner.
    main_code = (
        '"""Kaggle entry point for transformer_v2 (PairHead policy).\n'
        '\n'
        'Exposes ``agent(obs) -> moves`` so the Kaggle harness can ``import main``\n'
        'and call ``main.agent(...)``. The agent singleton is created lazily on\n'
        'the first call so module import stays cheap.\n'
        '"""\n'
        'from __future__ import annotations\n'
        '\n'
        'import os\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        '\n'
        'def _bundle_root() -> Path:\n'
        '    # Kaggle exec\'s main.py with no __file__. The harness chdirs to\n'
        '    # the agent dir and appends it to sys.path, so getcwd() and\n'
        '    # sys.path[-1] both point at the bundle root.\n'
        '    try:\n'
        '        here = Path(__file__).resolve().parent\n'
        '        if (here / "weights" / "entity_encoder_best.pt").exists():\n'
        '            return here\n'
        '    except NameError:\n'
        '        pass\n'
        '    cwd = Path(os.getcwd())\n'
        '    if (cwd / "weights" / "entity_encoder_best.pt").exists():\n'
        '        return cwd\n'
        '    for p in reversed(sys.path):\n'
        '        candidate = Path(p)\n'
        '        if (candidate / "weights" / "entity_encoder_best.pt").exists():\n'
        '            return candidate\n'
        '    fallback = Path("/kaggle_simulations/agent")\n'
        '    if (fallback / "weights" / "entity_encoder_best.pt").exists():\n'
        '        return fallback\n'
        '    raise FileNotFoundError(\n'
        '        "could not locate bundle root (no weights/entity_encoder_best.pt "\n'
        '        "found via __file__, cwd, sys.path, or /kaggle_simulations/agent)"\n'
        '    )\n'
        '\n'
        '\n'
        'HERE = _bundle_root()\n'
        '_HERE_STR = str(HERE)\n'
        '\n'
        '# Always prepend (do NOT guard on "not in"). The kaggle harness already\n'
        '# appended HERE at the END, but lux_ai_s3 also appended its own dir\n'
        '# (which contains a single-file ``agents.py``); without prepending we\n'
        '# load lux_ai_s3\'s ``agents.py`` instead of our package.\n'
        'sys.path.insert(0, _HERE_STR)\n'
        '\n'
        '# Purge any pre-existing ``agents`` module that resolves to something\n'
        '# OTHER than our bundle (e.g. lux_ai_s3 already imported its agents.py\n'
        '# eagerly). Leaving it in sys.modules makes the next import a no-op.\n'
        '_HERE_AGENTS = str((HERE / "agents").resolve())\n'
        'for _modname in [k for k in list(sys.modules)\n'
        '                 if k == "agents" or k.startswith("agents.")]:\n'
        '    _mod = sys.modules.get(_modname)\n'
        '    _mfile = getattr(_mod, "__file__", "") or ""\n'
        '    if not _mfile.startswith(_HERE_AGENTS):\n'
        '        del sys.modules[_modname]\n'
        '\n'
        'WEIGHTS_DIR = HERE / "weights"\n'
        'os.environ.setdefault(\n'
        '    "TRANSFORMER_V2_CKPT",\n'
        '    str(WEIGHTS_DIR / "entity_encoder_best.pt"),\n'
        ')\n'
        '\n'
        'from agents.transformer_v2.runner import TransformerAgent  # noqa: E402\n'
        '\n'
        '_AGENT: TransformerAgent | None = None\n'
        '\n'
        '\n'
        'def _ensure_agent() -> TransformerAgent:\n'
        '    global _AGENT\n'
        '    if _AGENT is None:\n'
        '        _AGENT = TransformerAgent.load(\n'
        '            ckpt_path=WEIGHTS_DIR / "entity_encoder_best.pt",\n'
        '            planet_run_dir=WEIGHTS_DIR,\n'
        '            fleet_run_dir=WEIGHTS_DIR,\n'
        '            comet_run_dir=WEIGHTS_DIR,\n'
        '        )\n'
        '    return _AGENT\n'
        '\n'
        '\n'
        'def agent(obs):\n'
        '    return _ensure_agent().act(obs)\n'
    )
    main_path = out_dir / "main.py"
    main_path.write_text(main_code)
    return main_path


def bundle_targz(bundle_dir: Path, archive_path: Path) -> Path:
    """tar+gzip the contents of ``bundle_dir`` into ``archive_path``.

    Archive uses top-level filenames (no leading dir) so Kaggle's extractor
    drops ``main.py`` + ``agents/`` + ``weights/`` directly at the root.
    """
    if archive_path.exists():
        archive_path.unlink()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in sorted(bundle_dir.iterdir()):
            tar.add(p, arcname=p.name)
    return archive_path


def kaggle_submit(
    archive: Path,
    *,
    note: str,
    competition: str = "orbit-wars",
    dry_run: bool = False,
) -> None:
    """Run ``kaggle competitions submit`` against ``archive``."""
    kaggle_bin = os.environ.get("KAGGLE_BIN") or shutil.which("kaggle")
    if not kaggle_bin:
        raise RuntimeError(
            "kaggle CLI not found. Install it (`pip install --user kaggle`) "
            "or set KAGGLE_BIN."
        )
    sha = _git_short_sha() + ("-dirty" if _git_dirty() else "")
    message = f"transformer_v2 @ {sha}"
    if note:
        message = f"{message} — {note}"

    cmd = [
        kaggle_bin, "competitions", "submit", competition,
        "-f", str(archive),
        "-m", message,
    ]
    print(f"[submit] file:    {archive}")
    print(f"[submit] message: {message}")
    if dry_run:
        print("[submit] dry-run:", " ".join(cmd))
        return

    # Kaggle CLI reads ``KAGGLE_KEY``; pass through ``KAGGLE_API_KEY`` as a
    # fallback so the user's .env (which uses KAGGLE_API_KEY) works.
    env = os.environ.copy()
    if "KAGGLE_KEY" not in env and "KAGGLE_API_KEY" in env:
        env["KAGGLE_KEY"] = env["KAGGLE_API_KEY"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle submit failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print("[submit] kaggle:", result.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "submissions" / "transformer_v2")
    ap.add_argument("--archive", type=Path,
                    default=REPO / "submissions" / "transformer_v2.tar.gz")
    ap.add_argument("--entity-ckpt", type=Path, default=DEFAULT_ENTITY_CKPT)
    ap.add_argument("--planet-ckpt", type=Path, default=DEFAULT_PLANET_CKPT)
    ap.add_argument("--fleet-ckpt", type=Path, default=DEFAULT_FLEET_CKPT)
    ap.add_argument("--comet-ckpt", type=Path, default=DEFAULT_COMET_CKPT)
    ap.add_argument("--submit", action="store_true",
                    help="Also run `kaggle competitions submit` against the bundle.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the kaggle submit command without executing.")
    ap.add_argument("--note", type=str, default="",
                    help="Free-text note appended to the submit message.")
    ap.add_argument("--competition", type=str, default="orbit-wars")
    args = ap.parse_args()

    build_bundle(
        out_dir=args.out_dir,
        entity_ckpt=args.entity_ckpt,
        planet_ckpt=args.planet_ckpt,
        fleet_ckpt=args.fleet_ckpt,
        comet_ckpt=args.comet_ckpt,
    )
    archive = bundle_targz(args.out_dir, args.archive)
    size_mb = archive.stat().st_size / 1e6
    print(f"[pack] bundle dir: {args.out_dir}")
    print(f"[pack] archive:    {archive} ({size_mb:.1f} MB)")

    if args.submit:
        kaggle_submit(
            archive, note=args.note,
            competition=args.competition, dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
