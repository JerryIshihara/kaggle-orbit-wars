"""Central actor<->learner state machine for distributed PPO (GCP VM + Colab).

``STATE.json`` on GCS / local, or an ``fs://`` Firestore doc, is the single
source of truth. Both sides act only on their phase: GCP rollout actor(s) on
``await_rollout``, the Colab learner on ``await_train``. A clean ping-pong; see
``docs/PPO_STATE_MACHINE.md`` for the full design + invariants.

Streaming
---------
STATE carries the active side's live ``progress`` plus a rolling ``log_tail``
(recent stdout lines) and a monotonic ``log_n`` (total lines ever emitted). The
owner rewrites STATE every couple seconds while working; watchers ``stream()`` it
with a cheap generation/etag skip (metadata-only when unchanged) and print only
the new lines -> the remote rollout/train streams to whoever is watching, over the
same GCS bus, ~1-2s latency, tiny transfers. No extra infra.

For TRUE push (no polling) swap the backend for a Firestore document + onSnapshot
listeners; this module's API is the seam, so it's a drop-in later.

Safety rests on a single invariant (details in the doc): exactly one side writes
STATE during each phase (phase gates ownership), and it uploads all artifacts
(shards+_DONE, or model pt) BEFORE flipping the phase -- so a referenced artifact
is always complete when the peer sees the new phase.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from . import shards

# --- phases --------------------------------------------------------------- #
PHASE_AWAIT_ROLLOUT = "await_rollout"   # model vK ready  -> rollout actor(s)
PHASE_ROLLING_OUT = "rolling_out"       # actor running (live progress)
PHASE_AWAIT_TRAIN = "await_train"       # shards vK ready -> LEARNER (Colab) trains
PHASE_TRAINING = "training"             # learner running (live progress)
PHASE_DONE = "done"
PHASE_ERROR = "error"

ACTOR = "actor"
LEARNER = "learner"

_LOG_TAIL_MAX = 60   # rolling stdout lines kept in STATE for streaming


class StatePhaseError(RuntimeError):
    """Raised when a transition's expected phase != the live phase (stale read /
    someone else moved it) so a confused side fails loud instead of double-acting."""


# --- Firestore backend (fs:// urls): push-streaming control channel -------- #
# Small/hot state (phase, iter, progress, log_tail) lives in a Firestore doc so
# both sides get pushed updates via on_snapshot (true streaming, no polling).
# Big/cold data (model .pt, shards) stays on GCS, referenced as gs:// urls in
# the doc. The google-cloud-firestore client is lazy-imported, so this module
# still loads on hosts without it (which only use gs:///local urls).
_FS_CLIENT = None


def _is_fs(url) -> bool:
    return str(url).startswith("fs://")


def _firestore_doc(url: str):
    """``fs://<collection>/<doc_id>`` -> a DocumentReference on the (default) db.
    Project from $PPO_FIRESTORE_PROJECT / $GOOGLE_CLOUD_PROJECT / ADC."""
    global _FS_CLIENT
    if _FS_CLIENT is None:
        from google.cloud import firestore
        proj = os.environ.get("PPO_FIRESTORE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        _FS_CLIENT = firestore.Client(project=proj) if proj else firestore.Client()
    path = str(url)[len("fs://"):].strip("/")
    coll, _, doc = path.partition("/")
    if not coll or not doc:
        raise ValueError(f"bad fs:// url {url!r}; expected fs://<collection>/<doc_id>")
    return _FS_CLIENT.collection(coll).document(doc)


def firestore_url(collection: str, doc_id: str) -> str:
    return f"fs://{collection}/{doc_id}"


# --- urls + raw io (local-or-GCS, via shards) ----------------------------- #
def state_url(prefix: str) -> str:
    return prefix.rstrip("/") + "/STATE.json"


def read_state(url: str) -> dict | None:
    """Current STATE, or None if absent. fs:// -> Firestore get; gs:///local ->
    shards.read_progress."""
    if _is_fs(url):
        snap = _firestore_doc(url).get()
        return snap.to_dict() if snap.exists else None
    return shards.read_progress(url)


def write_state(url: str, state: dict) -> None:
    """Full replace, stamping updated_ts. fs:// -> Firestore set (atomic doc
    write, pushed to all on_snapshot listeners); gs:// -> atomic tmp->cp."""
    st = dict(state)
    st["updated_ts"] = shards.utcnow_iso()
    if _is_fs(url):
        _firestore_doc(url).set(st)
        return
    shards.publish_progress(url, st)


def _empty_side() -> dict:
    return {"id": None, "claimed_iter": -1, "heartbeat_ts": None,
            "progress": None, "log_tail": [], "log_n": 0}


def init_state(url: str, *, run_id: str, iters_total: int, model: dict,
               config: dict, gcs_prefix: str | None = None,
               expected_actors: list[str] | None = None, message: str = "") -> dict:
    """Seed STATE at phase=await_rollout, iter=0 (call once at bootstrap).

    ``gcs_prefix`` is the gs:// run root for the BIG files (checkpoints/heads/
    rollouts). It lives in STATE so the fs:// control channel is fully decoupled
    from the gs:// data plane — daemons read paths from here, not from the
    (possibly fs://) state url.

    ``expected_actors`` is the list of actor-id strings (e.g.
    ``["c4-0", "n2-0"]``) that EACH roll a distinct, non-overlapping slice
    of episodes per iter; the phase only flips ``await_rollout -> await_train``
    once EVERY one has written its ``rollouts/vK/<id>/_DONE`` (a gated barrier).
    ``None`` = single-actor (the original ping-pong, back-compat). Per-actor live
    progress is mirrored under ``state['actors'][id]`` via ``update_actor`` so
    concurrent actors don't clobber one another."""
    st = {
        "run_id": run_id,
        "gcs_prefix": gcs_prefix,  # gs:// root for big files (checkpoints/heads/rollouts)
        "phase": PHASE_AWAIT_ROLLOUT,
        "iter": 0,
        "iters_total": int(iters_total),
        "model": model,            # {"heads":..., "full":..., "base_version":..., "policy_version":0}
        "shards": None,            # set to rollouts/vK/ at await_train (PARENT — union of all actors)
        "config": config,          # history_window, episodes, num_players, seed_base, device, ...
        "expected_actors": list(expected_actors) if expected_actors else None,
        ACTOR: _empty_side(),
        LEARNER: _empty_side(),
        "actors": {},              # per-actor {id: {progress, log_tail, log_n, status, heartbeat_ts}}
        "updated_by": "init",
        "message": message,
    }
    write_state(url, st)
    return st


def expected_actors(state: dict) -> list[str] | None:
    """The configured multi-actor roster (list of actor-id strings), or ``None``
    for the single-actor back-compat path."""
    a = (state or {}).get("expected_actors")
    return list(a) if a else None


def transition(url: str, *, expect_phase: str | None, new_phase: str,
               who: str, **fields) -> dict:
    """Read -> assert phase == expect_phase -> merge top-level ``fields`` -> set
    phase=new_phase -> write. The handoff write (rare). Pass expect_phase=None to
    force (e.g. writing ERROR/DONE)."""
    st = read_state(url)
    if st is None:
        raise StatePhaseError(f"no STATE at {url}")
    cur = st.get("phase")
    if expect_phase is not None and cur != expect_phase:
        raise StatePhaseError(
            f"expected phase {expect_phase!r} but STATE is {cur!r} "
            f"(iter {st.get('iter')}, updated_by {st.get('updated_by')})")
    st["phase"] = new_phase
    for k, v in fields.items():
        st[k] = v
    st["updated_by"] = who
    write_state(url, st)
    return st


def set_progress(url: str, side: str, progress: dict | None = None, *,
                 log_lines: list[str] | None = None, who: str | None = None) -> dict:
    """Cheap, frequent write by the ACTIVE owner while working: refresh its
    ``progress``, append new ``log_lines`` to the rolling tail (for streaming),
    bump ``log_n`` + ``heartbeat_ts``. The owner is the sole writer in its phase,
    so this read-modify-write is race-free."""
    st = read_state(url)
    if st is None:
        return {}
    blk = st.setdefault(side, _empty_side())
    if progress is not None:
        blk["progress"] = progress
    if log_lines:
        tail = list(blk.get("log_tail") or [])
        tail.extend(log_lines)
        blk["log_tail"] = tail[-_LOG_TAIL_MAX:]
        blk["log_n"] = int(blk.get("log_n", 0)) + len(log_lines)
    blk["heartbeat_ts"] = shards.utcnow_iso()
    if who:
        st["updated_by"] = who
    write_state(url, st)
    return st


def update_actor(url: str, actor_id: str, *, progress: dict | None = None,
                 log_lines: list[str] | None = None, status: str | None = None,
                 who: str | None = None) -> dict | None:
    """Per-actor live telemetry, written WITHOUT clobbering peer actors.

    In a multi-actor iter every actor is writing concurrently while in
    ``await_rollout``, so a naive read-modify-write of the whole STATE would race
    (last writer wins). We instead update only ``actors[actor_id]``:

      * fs:// (Firestore): a FIELD update via ``set({"actors": {actor_id: {...}}},
        merge=True)`` — ``merge=True`` does a DEEP merge of nested maps, so two
        actors writing ``actors.a`` and ``actors.b`` in the same doc never
        overwrite each other's sub-map (only the leaf fields each sets change).
      * gs:///local: best-effort read-modify-write of ``state['actors'][id]``
        (the single-writer-per-phase invariant is relaxed here, but each actor
        only touches its OWN sub-map, and a lost progress beat is harmless —
        the ``_DONE`` markers, not STATE, are the barrier ground truth).

    Keeps a rolling ``log_tail`` (last ``_LOG_TAIL_MAX``) + monotonic ``log_n``
    per actor, mirroring ``set_progress``."""
    ts = shards.utcnow_iso()
    if _is_fs(url):
        # Field-merge path: read just our own sub-map to roll the log tail, then
        # write back ONLY actors[actor_id] (deep-merged, peers untouched).
        prev = {}
        try:
            snap = _firestore_doc(url).get(["actors." + actor_id])
            if snap.exists:
                prev = ((snap.to_dict() or {}).get("actors") or {}).get(actor_id) or {}
        except Exception:
            prev = {}
        blk = _merge_actor_block(prev, progress=progress, log_lines=log_lines,
                                 status=status, ts=ts)
        _firestore_doc(url).set({"actors": {actor_id: blk}}, merge=True)
        return None
    # gs:///local: read-modify-write, but only our own actors[id] sub-map. We MERGE
    # this beat's delta onto the prior block (so an empty-log final flush keeps the
    # accumulated log_n/log_tail) — mirroring the fs:// deep-merge semantics.
    st = read_state(url)
    if st is None:
        return None
    actors = st.setdefault("actors", {})
    prev = dict(actors.get(actor_id) or {})
    delta = _merge_actor_block(prev, progress=progress, log_lines=log_lines,
                               status=status, ts=ts)
    prev.update(delta)
    actors[actor_id] = prev
    if who:
        st["updated_by"] = who
    write_state(url, st)
    return st


def _merge_actor_block(prev: dict, *, progress, log_lines, status, ts) -> dict:
    """Build the ``actors[id]`` DELTA for this beat from its previous value:
    refresh progress/status, append log_lines onto the rolling tail, bump log_n +
    heartbeat. Returns ONLY the keys that change (so the fs:// ``set(merge=True)``
    field-merge — and the local ``prev.update(delta)`` — touch the minimum and
    never drop the accumulated log on a beat that carries no new lines)."""
    blk: dict = {"heartbeat_ts": ts}
    if progress is not None:
        blk["progress"] = progress
    if status is not None:
        blk["status"] = status
    if log_lines:
        tail = list(prev.get("log_tail") or [])
        tail.extend(log_lines)
        blk["log_tail"] = tail[-_LOG_TAIL_MAX:]
        blk["log_n"] = int(prev.get("log_n", 0)) + len(log_lines)
    return blk


def claim_side(url: str, side: str, *, who: str, iter_: int) -> dict:
    """Stamp this side's id + claimed_iter (the lease). Single actor / single
    learner is trivially safe; the id matters only if a duplicate is launched."""
    st = read_state(url)
    blk = st.setdefault(side, _empty_side())
    blk["id"] = who
    blk["claimed_iter"] = int(iter_)
    blk["heartbeat_ts"] = shards.utcnow_iso()
    blk["progress"] = None
    write_state(url, st)
    return st


# --- idempotency helpers -------------------------------------------------- #
def done_marker_exists(shards_prefix: str) -> bool:
    """True if rollouts/vK/<id>/_DONE is already present (skip a redone rollout)."""
    return shards.gcs_exists(shards_prefix.rstrip("/") + "/_DONE")


def all_actors_done(gcs_prefix: str, policy_version: int,
                    actor_ids: list[str] | None) -> bool:
    """The multi-actor BARRIER: True iff every actor in ``actor_ids`` has written
    its ``{gcs_prefix}/rollouts/v{K}/<id>/_DONE`` marker for iter ``K`` (so the
    union of shards under the parent ``rollouts/v{K}/`` is complete and the phase
    may flip to ``await_train``).

    Returns ``False`` when ``actor_ids`` is None/empty — the single-actor path is
    handled separately by the caller (don't treat "no roster" as "all done")."""
    if not actor_ids:
        return False
    base = gcs_prefix.rstrip("/") + f"/rollouts/v{int(policy_version)}/"
    return all(done_marker_exists(base + aid + "/") for aid in actor_ids)


def model_block(prefix: str, policy_version: int, base_version: str) -> dict:
    """The ``model`` block for iter K: where the actor pulls the head-delta from."""
    p = prefix.rstrip("/")
    return {
        "heads": f"{p}/heads/policy_v{policy_version}.heads.pt",
        "full": f"{p}/checkpoints/policy_v{policy_version}.pt",
        "base_version": base_version,
        "policy_version": int(policy_version),
    }


def gcs_prefix(state: dict) -> str | None:
    """The gs:// run root for big files (checkpoints/heads/rollouts), set at
    init_state. Daemons build big-file paths from THIS, not from the state url
    (which may be fs://). Falls back to the state url if it is itself gs://."""
    p = (state or {}).get("gcs_prefix")
    if p:
        return str(p).rstrip("/")
    return None


# --- streaming watcher ---------------------------------------------------- #
def _fs_stream(url, *, until_phases, watch_side, on_state, on_log, stall_s):
    """``stream()`` for fs:// urls: TRUE push via Firestore on_snapshot (no
    polling). A background listener pushes doc snapshots onto a queue; we drain
    them and emit on_state / on_log exactly like the poll path."""
    import queue as _queue

    until = set(until_phases) if isinstance(until_phases, (set, list, tuple)) else {until_phases}
    doc = _firestore_doc(url)
    q: _queue.Queue = _queue.Queue()

    def _cb(snaps, changes, read_time):
        for s in snaps:
            d = s.to_dict()
            if d is not None:
                q.put(d)

    watch = doc.on_snapshot(_cb)
    seen_n = 0
    last_change = time.time()
    final = None
    try:
        while True:
            try:
                st = q.get(timeout=stall_s if stall_s else 30.0)
            except _queue.Empty:
                if stall_s and (time.time() - last_change) > stall_s:
                    raise TimeoutError(
                        f"Firestore doc {url} unchanged for {stall_s:.0f}s — peer stalled?")
                continue
            last_change = time.time()
            if on_state:
                on_state(st)
            if on_log and watch_side:
                blk = st.get(watch_side) or {}
                log_n = int(blk.get("log_n", 0))
                if log_n > seen_n:
                    tail = blk.get("log_tail") or []
                    on_log(tail[-min(log_n - seen_n, len(tail)):])
                    seen_n = log_n
            ph = st.get("phase")
            if ph in until or ph == PHASE_ERROR:
                final = st
                break
    finally:
        watch.unsubscribe()
    return final


def stream(url: str, *, until_phases, poll_s: float = 1.5, watch_side: str | None = None,
           on_state: Callable[[dict], None] | None = None,
           on_log: Callable[[list[str]], None] | None = None,
           stall_s: float | None = None) -> dict:
    """Fast-poll STATE until ``phase in until_phases`` (or ERROR), emitting
    streaming updates so the caller watches the remote side live:

      - ``on_state(state)`` fires whenever STATE content changes.
      - ``on_log(new_lines)`` fires with the NEW ``log_tail`` lines from
        ``watch_side`` (deduped via ``log_n``) -> a continuous tail of the remote
        rollout/train stdout.

    Returns the final state. Raises ``TimeoutError`` if STATE is unchanged for
    ``stall_s`` (dead peer). For fs:// urls this is TRUE PUSH (Firestore
    on_snapshot); for gs:///local it fast-polls. Callers don't change.
    """
    if _is_fs(url):
        return _fs_stream(url, until_phases=until_phases, watch_side=watch_side,
                          on_state=on_state, on_log=on_log, stall_s=stall_s)
    until = set(until_phases) if isinstance(until_phases, (set, list, tuple)) else {until_phases}
    seen_n = 0
    last_repr = None
    last_change = time.time()
    while True:
        st = read_state(url)
        if st is not None:
            rep = repr(st)
            if rep != last_repr:
                last_repr = rep
                last_change = time.time()
                if on_state:
                    on_state(st)
                if on_log and watch_side:
                    blk = st.get(watch_side) or {}
                    log_n = int(blk.get("log_n", 0))
                    if log_n > seen_n:
                        tail = blk.get("log_tail") or []
                        new = log_n - seen_n
                        on_log(tail[-min(new, len(tail)):])   # ring may drop if far behind; full log in rollout.log
                        seen_n = log_n
            ph = st.get("phase")
            if ph in until or ph == PHASE_ERROR:
                return st
        if stall_s and (time.time() - last_change) > stall_s:
            raise TimeoutError(
                f"STATE unchanged for {stall_s:.0f}s (phase "
                f"{st.get('phase') if st else '??'}) — peer stalled?")
        time.sleep(poll_s)
