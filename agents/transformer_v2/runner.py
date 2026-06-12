"""UI-runnable agent: 2-head PairHead stack loaded from a v2 entity-pretrain ckpt.

Loads an ``entity_encoder_best.pt`` produced by
``agents/transformer_v2/pretrain/entity_encoder.py``, instantiates the
full L1-L4 + PairHead stack, and runs inference per tick:

  1. Featurize the current obs via :func:`featurize_observation`
     (single-frame; T-history is not maintained at inference time, the
     model accepts T=1 input via L2's step-embedding tail slice).
  2. Run L0 frozen specialists on the raw features → ``planet_tok``,
     ``comet_tok``, ``fleet_tok``.
  3. ``where(is_comet, comet_tok, planet_tok)`` → ``entity_self``.
  4. Forward through :class:`EntityPretrainModel` → ``pair_logits`` and
     ``pair_frac``.
  5. Mask ``pair_logits`` by per-row source-launchability and per-column
     target-validity; emit every off-diagonal cell whose logit clears
     the configured threshold.
  6. Use ``sigmoid(pair_frac[src, tgt])`` as a fraction of the source
     planet's ships, with a per-source budget so multi-target rows do
     not double-spend.
  7. Validate each chosen pair through :func:`physics_utils.plan_launch`
     before emitting the move, so wrong-planet / sun / boundary launches
     are dropped instead of sent to the env.

Default ckpt resolution (override with ``TRANSFORMER_V2_CKPT`` env var
or :meth:`TransformerAgent.load`'s ``ckpt_path`` arg): newest
``entity_encoder_best.pt`` under ``data/runs/entity/<run>/``.

The L0 frozen specialists default to the d=256 ckpts under
``data/runs/{planet,fleet,comet}/specialist_*``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from ..heuristic.physical_v4.agent import (
    PHASE_TABLE,
    candidates_for_source,
    compute_surplus,
    infer_rotation_sign,
    phase_of,
)
from ..physics_utils import (
    P_ID,
    _infer_rotation_sign_raw,
    plan_launch,
)
from ..registry import register
from .featurizer import FleetTracker, featurize_observation
from .paths import ENTITY_RUNS_DIR
from .pretrain.entity_encoder import (
    EntityPretrainModel,
    _build_entity_self_tokens,
    _load_encoders,
    build_pair_type_ids,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANET_RUN_DIR = _REPO_ROOT / "data" / "runs" / "planet" / "specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep"
DEFAULT_FLEET_RUN_DIR = _REPO_ROOT / "data" / "runs" / "fleet" / "specialist_fleet_d256_40k_lr1e4_120ep"
DEFAULT_COMET_RUN_DIR = _REPO_ROOT / "data" / "runs" / "comet" / "fullpath_scalar_multitask_d256_40k_lr1e4_120ep"


def _default_ckpt() -> Path:
    """Pick the newest ``entity_encoder_best.pt`` (or ``entity_encoder_last.pt``)
    under ``data/runs/entity/<run>/``, ranked by directory mtime. Override
    with the ``TRANSFORMER_V2_CKPT`` env var.
    """
    env_override = os.environ.get("TRANSFORMER_V2_CKPT")
    if env_override:
        return Path(env_override)
    if not ENTITY_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"no entity runs dir at {ENTITY_RUNS_DIR}; train an "
            "entity-pretrain model first or set TRANSFORMER_V2_CKPT "
            "to a saved ckpt path."
        )
    run_dirs = sorted(
        (d for d in ENTITY_RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for name in ("entity_encoder_best.pt", "entity_encoder_last.pt"):
            p = run_dir / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"no entity_encoder_*.pt under {ENTITY_RUNS_DIR}/*/."
    )


def _validated_learner_launch(
    *,
    source_raw,
    target_raw,
    ships: int,
    planets: list,
    fleets: list,
    initial_planets: list,
    angular_velocity: float,
    comet_ids: set[int],
    comets: list,
    player: int,
    surplus: int,
    safety_buffer: int,
    current_step: int,
) -> tuple[float, float] | None:
    """Return ``(angle, eta)`` only if the learned launch is env-valid.

    The old v2 path used the low-level ``shoot_*`` helpers directly and
    fell back to naive ``atan2`` on errors. Those helpers intentionally
    emit an angle even when the first collision is the sun, board
    boundary, or a different planet. For learned pair selection that is
    too permissive: a bad top pair becomes a real miss/out-of-map move.

    This function uses the high-level planner instead. It pins the
    learned ship count, validates first collision against moving planets
    / comets, and returns ``None`` unless the trajectory's first planet
    hit is exactly the intended target.
    """
    if source_raw is None or target_raw is None or ships <= 0:
        return None
    try:
        target_id = int(target_raw[P_ID])
        av_sign = _infer_rotation_sign_raw(planets, initial_planets)
        launch = plan_launch(
            source_raw,
            target_raw,
            planets=planets,
            fleets=fleets,
            player=player,
            angular_velocity=angular_velocity,
            av_sign=av_sign,
            comet_planet_ids=comet_ids,
            comets=comets,
            fleet_ships=int(ships),
            surplus=int(surplus),
            safety_buffer=safety_buffer,
            current_step=current_step,
        )
        if not launch.ok or launch.actual_hit_id != target_id:
            return None
        return float(launch.angle), float(launch.eta)
    except Exception:
        return None


class TransformerAgent:
    """Inference wrapper around the v2 entity-pretrain PairHead policy."""

    #: Valid values for ``inference_mode`` — controls how ``_predict`` turns
    #: ``pair_logits`` into a launch list. ``threshold`` is the production
    #: rule for sigmoid-frac checkpoints; ``alloc_softmax`` is the production
    #: rule for ``bernoulli_select_multinomial_alloc_v2`` checkpoints (same
    #: fired set, sizes from the contract's share softmax incl. the HOLD
    #: diagonal); the others are diagnostic alternatives.
    INFERENCE_MODES = (
        "threshold", "alloc_softmax", "topk_self", "row_argmax", "flat_argmax",
        "single_target",
    )

    def __init__(
        self,
        model: EntityPretrainModel,
        fleet_enc,
        planet_enc,
        comet_enc,
        *,
        device: str = "cpu",
        deterministic: bool = True,
        max_planets: int = 64,
        max_fleets: int = 256,
        num_players: int = 4,
        inference_mode: str = "threshold",
        logit_threshold: float = 2.0,
        select_k_max: int = 3,
    ):
        if inference_mode not in self.INFERENCE_MODES:
            raise ValueError(
                f"inference_mode={inference_mode!r} not in {self.INFERENCE_MODES}"
            )
        self.model = model.to(device).eval()
        self.fleet_enc = fleet_enc.to(device).eval()
        self.planet_enc = planet_enc.to(device).eval()
        self.comet_enc = comet_enc.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.num_players = num_players
        self.inference_mode = inference_mode
        self.logit_threshold = float(logit_threshold)
        self.select_k_max = int(select_k_max)
        # Per-episode launch-tick state. Reset on step=0 detection.
        self._tracker = FleetTracker()
        self._last_step: int | None = None

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path | None = None,
        *,
        device: str | None = None,
        deterministic: bool = True,
        planet_run_dir: Path | None = None,
        fleet_run_dir: Path | None = None,
        comet_run_dir: Path | None = None,
        inference_mode: str | None = None,
        logit_threshold: float = 2.0,
        disable_film: bool = False,
    ) -> "TransformerAgent":
        """Reconstruct the L0 frozen specialists + the trainable
        L1-L4 + PairHead stack from a v2 entity-pretrain ckpt.

        ``disable_film=True`` flips the PairHead's runtime FiLM bypass on
        after loading — the trained FiLM weights stay in the ckpt but the
        modulation is skipped at inference (h_film = h). Used to A/B the
        FiLM contribution head-to-head on identical weights.

        ``ckpt_path`` defaults to the newest ``entity_encoder_best.pt``
        under ``data/runs/entity/<run>/``. The L0 ckpt dirs default to
        the d=256 specialists shipped with the repo.
        """
        ckpt_path = Path(ckpt_path) if ckpt_path is not None else _default_ckpt()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"transformer_v2 ckpt not found at {ckpt_path}. "
                "Train an entity-pretrain model first or set "
                "TRANSFORMER_V2_CKPT."
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config") or {}
        # Auto-select the decision rule from the ckpt's action contract when the
        # caller did not pin one. A ``single_target_per_source_v1`` ckpt (joint /
        # single-target pretrain) MUST decode per-source argmax over
        # [targets + diagonal-NOOP]; using the default ``threshold`` rule on it
        # silently mis-decodes (the over-holding diagnosis). Legacy ckpts have no
        # ``action_contract`` and fall back to ``threshold`` exactly as before.
        if inference_mode is None:
            contract = cfg.get("action_contract")
            if contract == "single_target_per_source_v1":
                inference_mode = "single_target"
            elif contract == "bernoulli_select_multinomial_alloc_v2":
                # v2 pretrains train frac_loc as softmax-SHARE logits (incl.
                # the HOLD diagonal); sigmoid sizing would mis-read them.
                inference_mode = "alloc_softmax"
            elif contract == "bounded_k_select_multinomial_alloc_v3":
                # v3: self logit = learned firing threshold; floor + extras.
                inference_mode = "topk_self"
            elif contract == "bounded_k_select_dirichlet_alloc_v4":
                # v4: same bounded-k select; alloc mean = the SAME frac
                # softmax (Dirichlet only changes the sampled spread), so
                # every topk_self decode arm applies unchanged. Deterministic
                # deploy = OW_V3_DECODE=expmatch (mean-share sizing).
                inference_mode = "topk_self"
            else:
                inference_mode = "threshold"
            print(f"[runner] inference_mode auto-selected: {inference_mode!r} "
                  f"(action_contract={contract!r})", flush=True)
        d_model = int(cfg.get("d_model", 256))
        n_steps = int(cfg.get("n_steps", 6))
        # Older ckpts (pre-d_pair widening) did not save ``d_pair`` and
        # hardcoded PairHead's projection width to 128. New runs save the
        # explicit d_pair into the config (defaulting to d_model when
        # unspecified). Read it back here so both old and new ckpts
        # deserialize correctly.
        d_pair = int(cfg["d_pair"]) if "d_pair" in cfg else 128
        # Head counts. Pre-standardization ckpts hardcoded L1 / L3 / L4 to
        # 4 heads and L2 to 8 heads. The current line saves the configured
        # head counts so both layouts deserialize correctly. State_dict
        # shapes are head-count-independent (MultiheadAttention stores
        # ``in_proj_weight (3*d_model, d_model)`` and ``out_proj`` at the
        # full width), so the keys load fine — but the FORWARD numerics
        # change if you mis-pair weights with a different head count. The
        # fallbacks here mirror the pre-standardization defaults.
        entity_n_heads = int(cfg["entity_n_heads"]) if "entity_n_heads" in cfg else 4
        cross_n_heads = int(cfg.get("cross_n_heads", 8))
        cross_n_layers = int(cfg.get("cross_n_layers", 2))
        dual_n_heads = int(cfg["dual_n_heads"]) if "dual_n_heads" in cfg else 4
        # FiLM conditioner depth. Pre-deepening ckpts saved no field;
        # fall back to 1 (the original 2-Linear conditioner). The shim
        # in PairHead._prune_legacy_state_dict will drop the film_proj
        # weights if depth changed, so an old-depth ckpt loaded into a
        # new-depth module degrades to identity-init FiLM rather than
        # blowing up on key-shape mismatch.
        conditioner_n_layers = int(cfg.get("conditioner_n_layers", 1))
        # Per-head decoder MLP depth. Default 1 = legacy single Linear.
        # The shim in PairHead._prune_legacy_state_dict drops the head
        # weights if the saved depth doesn't match the new instantiation,
        # so old single-Linear ckpts loaded into a deeper-MLP module
        # leave the heads at random init rather than mid-loading shapes.
        head_n_layers = int(cfg.get("head_n_layers", 1))
        # Architecture flags (skip_l34 / with_consolidator). Prefer the saved
        # config, but older ablation ckpts (e.g. the noL34 run) did NOT save
        # these — so DETECT from the state_dict keys: no dual_role/joint_role
        # keys ⇒ skip_l34; no consolidator keys ⇒ with_consolidator=False.
        # Without this, a skip-L3/L4 ckpt would build a full model and leave
        # L3/L4 at RANDOM INIT (the PairHead would then read garbage), making
        # the loaded agent silently broken.
        _sd_keys = ckpt["model"].keys()
        if "skip_l34" in cfg:
            skip_l34 = bool(cfg["skip_l34"])
        else:
            skip_l34 = not any(
                k.startswith("dual_role.") or k.startswith("joint_role.")
                for k in _sd_keys
            )
        if "with_consolidator" in cfg:
            with_consolidator = bool(cfg["with_consolidator"])
        else:
            with_consolidator = any(k.startswith("consolidator.") for k in _sd_keys)

        # Load L0 frozen specialists by their run dirs. _load_encoders
        # reads each ckpt's config for d_model + use_traj_branch.
        fleet_enc, planet_enc, comet_enc = _load_encoders(
            fleet_run_dir or DEFAULT_FLEET_RUN_DIR,
            planet_run_dir or DEFAULT_PLANET_RUN_DIR,
            comet_run_dir or DEFAULT_COMET_RUN_DIR,
            device=device,
        )

        arch = str(cfg.get("arch", "v2"))
        if arch == "dual_rate_l2_v3":
            # transformer_v3 dual-rate L2 (v3.1 player tokens, no
            # consolidator). Single-frame inference works natively: both
            # branches see the frame as T=1 and the zero-/trained fusion
            # applies — smoke-proven parity in transformer_v3.
            from ..transformer_v3.model import EntityPretrainModelV3
            model = EntityPretrainModelV3(
                d_model=d_model, d_pair=d_pair,
                entity_n_heads=entity_n_heads,
                cross_n_heads=cross_n_heads,
                cross_n_layers=cross_n_layers,
                dual_n_heads=dual_n_heads,
                conditioner_n_layers=conditioner_n_layers,
                head_n_layers=head_n_layers,
                skip_l34=skip_l34,
                with_consolidator=False, with_value_heads=False,
                with_short_aux=bool(cfg.get("with_short_aux", False)),
                with_alloc_conc=bool(cfg.get("with_alloc_conc", False)),
            )
        else:
            model = EntityPretrainModel(
                d_model=d_model, n_steps=n_steps, d_pair=d_pair,
                entity_n_heads=entity_n_heads,
                cross_n_heads=cross_n_heads,
                cross_n_layers=cross_n_layers,
                dual_n_heads=dual_n_heads,
                conditioner_n_layers=conditioner_n_layers,
                head_n_layers=head_n_layers,
                skip_l34=skip_l34,
                with_consolidator=with_consolidator,
            )
        # ``strict=False`` so legacy 5-head ckpts (with ``source_act_head`` /
        # ``target_aim_head`` / ``glob_act_head`` keys and no ``film_proj`` /
        # ``film_alpha`` keys) still load. The unexpected legacy aux-head
        # weights are dropped; missing FiLM keys fall back to the
        # identity-init defaults (γ=β=0, film_alpha=1 → FiLM is a no-op
        # at start, but trainable immediately; old ckpt behavior is
        # preserved bit-for-bit on the pair_logits / pair_frac heads).
        model.load_state_dict(ckpt["model"], strict=False)

        # Inference-time FiLM ablation toggle (does not affect loaded weights).
        if disable_film:
            model.pair_head.disable_film = True

        return cls(
            model,
            fleet_enc=fleet_enc,
            planet_enc=planet_enc,
            comet_enc=comet_enc,
            device=device,
            deterministic=deterministic,
            max_planets=cfg.get("max_planets", 64),
            max_fleets=cfg.get("max_fleets", 1024),
            inference_mode=inference_mode,
            logit_threshold=logit_threshold,
            select_k_max=int(cfg.get("select_k_max", 3)),
        )

    @torch.no_grad()
    def _predict(
        self,
        obs: dict[str, Any],
        learner_slot: int,
        *,
        logit_threshold: float | None = None,
        inference_mode: str | None = None,
    ) -> list[tuple[int, int, float]]:
        """Return ``(source_pid, target_pid, frac)`` triples for this turn.

        The decision rule is chosen by ``inference_mode``:

          * ``threshold`` *(default — production rule)*: every legal
            ``(s, t)`` cell whose ``pair_logits[s, t] > logit_threshold``
            fires, producing potentially multiple launches per source.
            Multi-target / coalition behavior falls out for free.
          * ``row_argmax``: for each launchable source ``s``, pick the
            single best target via ``argmax_t pair_logits[s, t]``. At most
            one launch per launchable source per turn.
          * ``flat_argmax``: pick the single best ``(s, t)`` over the full
            P×P grid. At most one launch per turn (the legacy single-shot
            rule from before the threshold rewrite).

        Self-cells (``s == t``) and padded slots are always dropped — the
        diagonal was never supervised, and pad rows are uncalibrated.
        Each emitted launch is sized by ``sigmoid(pair_frac[s, t])`` of
        the source's ships, with per-source surplus budgeting applied in
        ``act``.

        Keyword args override the per-instance defaults; callers that
        want to A/B inference modes on the same loaded model can pass
        them directly rather than constructing multiple agents.
        """
        if inference_mode is None:
            inference_mode = self.inference_mode
        if logit_threshold is None:
            logit_threshold = self.logit_threshold
        if inference_mode not in self.INFERENCE_MODES:
            raise ValueError(
                f"inference_mode={inference_mode!r} not in {self.INFERENCE_MODES}"
            )
        # Reset launch-tick state at episode start.
        step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
        if self._last_step is None or step < self._last_step:
            self._tracker = FleetTracker()
        self._last_step = step

        # Single-frame featurization. The model was trained with T=6
        # history but accepts T=1 via L2's step_embed[-T:] tail slice;
        # the loss in fidelity is bounded by the L2 attention's
        # step-position generalization.
        batch, pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=self._tracker,
            num_players=self.num_players,
            max_planets=self.max_planets,
            max_fleets=self.max_fleets,
            device=self.device,
        )

        # Need comet_features too — the inference featurizer doesn't
        # emit it (training-only key on the cached snapshots). Stub it
        # with zeros for now; the comet_tok pathway will just produce
        # zero-encoded tokens for comet planets, which `where` routes
        # away from for non-comet planets anyway. is_comet is taken
        # from f000 of planet_features per the comet specialist's input
        # convention.
        B, P, _ = batch["planet_features"].shape
        comet_input_dim = self.comet_enc.input_dim
        comet_features = torch.zeros(
            (B, P, comet_input_dim), device=self.device,
            dtype=batch["planet_features"].dtype,
        )
        # The first 18 dims of the planet feature vector are the same
        # scalar block the comet specialist saw (f000 = is_comet flag,
        # f001..f017 = sun-relative geometry + owner one-hot + ships).
        comet_features[..., :18] = batch["planet_features"][..., :18]

        # is_comet flag: f000 of the planet feature vector is the
        # is_comet flag in the v2 featurizer.
        is_comet = batch["planet_features"][..., 0] > 0.5

        # L0 frozen forward.
        planet_tok = self.planet_enc(batch["planet_features"])     # (B, P, d)
        comet_tok = self.comet_enc(comet_features)                  # (B, P, d)
        fleet_tok = self.fleet_enc(batch["fleet_features"])         # (B, F, d)
        entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)

        routing = {
            "fleet_target_idx": batch["fleet_target_idx"],
            "fleet_source_idx": batch["fleet_source_idx"],
            "fleet_owner_slot": batch["fleet_owner_slot"],
            "fleet_ships_log": batch["fleet_ships_log"],
            "fleet_eta_norm": batch["fleet_eta_norm"],
            "fleet_mask": batch["fleet_mask"],
        }
        if getattr(self.model, "ARCH", "") == "dual_rate_l2_v3":
            # v3.1: the trained owner projection must participate at
            # inference (it's zero-init only at the START of training) —
            # route through forward_with_context, which threads the
            # learner-relative owner one-hot into the dual L2. Returns the
            # same pair_logits / pair_frac keys.
            from .pretrain.entity_encoder import (
                ENTITY_N_OWNER_CLASSES as _N_OWN,
                _PLANET_OWNER_START_IDX as _OWN0,
            )
            preds = self.model.forward_with_context(
                entity_self, fleet_tok, routing, batch["planet_mask"],
                is_comet=is_comet,
                pair_type_ids=build_pair_type_ids(
                    batch["planet_features"], batch["planet_mask"],
                ),
                planet_owner_oh=batch["planet_features"][
                    ..., _OWN0:_OWN0 + _N_OWN],
            )
        else:
            preds = self.model(
                entity_self, fleet_tok, routing, batch["planet_mask"],
                is_comet=is_comet,
                pair_type_ids=build_pair_type_ids(
                    batch["planet_features"], batch["planet_mask"],
                ),
            )

        pair_logits = preds["pair_logits"].squeeze(0)               # (P, P)
        pair_frac_raw = preds["pair_frac"].squeeze(0)               # (P, P) raw logit
        idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}

        # ---- Build launchable / ownership masks from obs ----
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        _neutral_bonus, defense_buffer, min_launch, _safety, _frontier_w, _eta_tol = (
            PHASE_TABLE[phase_of(step)]
        )
        owner_by_idx: dict[int, int] = {}
        launchable_by_idx: dict[int, bool] = {}
        ships_by_idx: dict[int, int] = {}
        if raw_planets:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
            for planet in planets:
                pid = int(planet.id)
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    owner = int(planet.owner)
                    owner_by_idx[idx] = owner
                    ships_by_idx[idx] = int(planet.ships)
                    surplus = compute_surplus(planet, enemy_fleets, defense_buffer)
                    launchable_by_idx[idx] = (
                        owner == learner_slot and surplus >= min_launch
                    )

        P = pair_logits.size(0)
        # ``real_idx[i]`` = there's a real planet (not padding) at slot i.
        # Padded slots were masked out of ``pair_valid`` during training,
        # so their logits are uncalibrated noise — drop them.
        real_idx = torch.zeros(P, dtype=torch.bool, device=pair_logits.device)
        src_legal = torch.zeros(P, dtype=torch.bool, device=pair_logits.device)
        for i in range(P):
            if i not in idx_to_pid:
                continue
            real_idx[i] = True
            if launchable_by_idx.get(i, False):
                src_legal[i] = True

        if not src_legal.any():
            return []

        # Build the legality mask once; all three inference modes share it.
        # Legality: source launchable, target real, t != s (diagonal was
        # unsupervised — exclude).
        device = pair_logits.device
        eye = torch.eye(P, dtype=torch.bool, device=device)
        cell_legal = (
            src_legal.unsqueeze(1)               # (P, 1) — source must be launchable
            & real_idx.unsqueeze(0)              # (1, P) — target must be real
            & ~eye                                # exclude self-cells (off-diagonal only)
        )
        if not cell_legal.any():
            return []

        neg_inf = torch.finfo(pair_logits.dtype).min
        masked_logits = pair_logits.masked_fill(~cell_legal, neg_inf)

        # --- Dispatch on inference_mode ---
        if inference_mode == "flat_argmax":
            # Single-shot: pick the highest legal cell across the full grid.
            flat_idx = int(masked_logits.reshape(-1).argmax().item())
            if masked_logits.reshape(-1)[flat_idx].item() == neg_inf:
                return []
            src_indices = torch.tensor([flat_idx // P], device=device)
            tgt_indices = torch.tensor([flat_idx % P],  device=device)
        elif inference_mode == "row_argmax":
            # One launch per launchable source — argmax over its row.
            row_best = masked_logits.argmax(dim=-1)                    # (P,)
            keep = src_legal & (
                masked_logits.gather(1, row_best.unsqueeze(-1)).squeeze(-1) > neg_inf
            )
            src_indices = torch.nonzero(keep, as_tuple=False).flatten()
            tgt_indices = row_best[src_indices]
        elif inference_mode == "single_target":
            # single_target_per_source_v1 contract: each launchable source
            # picks the argmax over [legal off-diagonal targets PLUS its own
            # diagonal == NOOP/hold]. If the diagonal wins, the source HOLDS
            # (no launch). Requires a model trained with the single-target CE
            # so pair_logits[s, s] is the calibrated hold logit.
            diag = torch.arange(P, device=device)
            row_legal = cell_legal.clone()
            row_legal[diag, diag] = src_legal          # add the hold slot per launchable src
            row_masked = pair_logits.masked_fill(~row_legal, neg_inf)
            row_best = row_masked.argmax(dim=-1)                       # (P,) may be the diagonal
            keep = (
                src_legal
                & (row_best != diag)                  # diagonal winning == hold -> drop
                & (row_masked.gather(1, row_best.unsqueeze(-1)).squeeze(-1) > neg_inf)
            )
            src_indices = torch.nonzero(keep, as_tuple=False).flatten()
            tgt_indices = row_best[src_indices]
            # Opt-in decode diagnostics: how many launchable sources HELD
            # (diagonal/NOOP won the row) vs LAUNCHED this turn. Set
            # ``agent._debug_decode = True`` to accumulate into ``agent._dbg``.
            if getattr(self, "_debug_decode", False):
                dbg = self.__dict__.setdefault(
                    "_dbg", {"turns": 0, "src_legal": 0, "launched": 0, "held": 0}
                )
                nsl = int(src_legal.sum())
                nl = int(keep.sum())
                dbg["turns"] += 1
                dbg["src_legal"] += nsl
                dbg["launched"] += nl
                dbg["held"] += nsl - nl
        elif inference_mode == "topk_self":
            # bounded_k v3 deploy decode. Per launchable row, the SELF logit
            # ``pair_logits[s, s]`` is the LEARNED firing threshold: fire every
            # legal target whose select logit beats it, capped at
            # ``k = min(select_k_max, ships // min_launch)`` by descending
            # logit. Sizes follow the v3 allocation exactly: ``min_launch +
            # softmax([frac_loc[s, F], frac_loc[s, s]]) · remainder``. No
            # constant threshold anywhere; act()'s surplus budget still caps.
            picked: list[tuple[float, int, int, float]] = []
            for s in range(P):
                if not bool(src_legal[s]):
                    continue
                n_ships = int(ships_by_idx.get(s, 0))
                k_cap = min(int(self.select_k_max),
                            n_ships // max(1, int(min_launch)))
                if k_cap <= 0:
                    continue
                row = masked_logits[s]                       # -inf on illegal
                self_logit = float(pair_logits[s, s])
                # Default expmatch@0.5: the 9-update matrix measured guard@0.5
                # 21.9% / lifted@0.5 25.0% / threshold-free 6.2% — deterministic
                # per-step decoding needs the marginal bar as a CADENCE damper
                # (sampling fires a p-target every ~1/p steps; threshold-free
                # fires it EVERY step).
                decode = os.environ.get("OW_V3_DECODE", "sample")
                if decode == "sample":
                    # SAMPLED deploy (production default): draw the action
                    # from the v3 contract EXACTLY as training does — no bar,
                    # no decode rule, no train/deploy gap. The deployed agent
                    # IS the optimized policy; deploy strength tracks sampled
                    # strength 1:1 as the model learns. (Determinism was an
                    # assumption inherited from the threshold era, not a
                    # competition requirement.)
                    from .ppo.sampler import sample_bounded_k
                    n_legal_t = int((row > -1e30).sum())
                    if n_legal_t == 0:
                        continue
                    pm_row = torch.zeros(P, P, dtype=torch.bool,
                                         device=row.device)
                    pm_row[s] = row > -1e30
                    sm_row = torch.zeros(P, dtype=torch.bool,
                                         device=row.device)
                    sm_row[s] = True
                    ships_vec = torch.zeros(P, dtype=torch.long,
                                            device=row.device)
                    ships_vec[s] = n_ships
                    act_s = sample_bounded_k(
                        pair_logits, pair_frac_raw, ships_vec,
                        pair_mask=pm_row, source_mask=sm_row,
                        min_launch=int(min_launch),
                        k_max=int(self.select_k_max),
                    )
                    fired_cols = (act_s.select_counts[s, :P] >= 1).nonzero(
                        as_tuple=False).flatten().tolist()
                    for t in fired_cols:
                        ships_t = int(min_launch) + int(
                            act_s.alloc_extras[s, t].item())
                        picked.append((float(row[t]), s, int(t),
                                       ships_t / max(1, n_ships)))
                    continue
                if decode == "expcount":
                    # THRESHOLD-FREE decode (production default): the select
                    # distribution's own structure sets firing breadth. The
                    # sampler draws k tokens from softmax([legal, self]); in
                    # expectation k*(1-p_self) of them land on targets, so
                    # fire the top round(k*(1-p_self)) targets by probability.
                    # Hedged rows fire (mass on targets even when no single
                    # target dominates); holdy rows hold (self mass -> 0).
                    # No tunable constant anywhere.
                    cat_logits = torch.cat([
                        row, torch.tensor([self_logit], device=row.device),
                    ])
                    probs = torch.softmax(cat_logits, dim=0)
                    n_fire = min(k_cap, int(round(
                        k_cap * float(1.0 - probs[-1]))))
                    if n_fire <= 0:
                        continue                             # hold
                    tp = probs[:P]
                    order_t = torch.argsort(tp, descending=True)[:n_fire]
                    order_t = order_t[tp[order_t] > 0]       # drop illegal (p=0)
                    if int(order_t.numel()) == 0:
                        continue
                elif decode == "expmatch":
                    # Expectation-matched firing: reproduce the SAMPLER's
                    # marginals instead of thresholding on the raw self
                    # logit. Sampling fires t when >=1 of k draws from
                    # softmax([legal, self]) lands on t — P(fire t) =
                    # 1-(1-p_t)^k. Fire the cells where that marginal >= 0.5.
                    # Robust to a noisy self logit (it is one normalizer term
                    # among ~30, exactly as in training, rather than the sole
                    # gatekeeper — the diagonal was DEAD in the v2 pretrain
                    # and is only partially calibrated early in v3 PPO).
                    cat_logits = torch.cat([
                        row, torch.tensor([self_logit], device=row.device),
                    ])
                    probs = torch.softmax(cat_logits, dim=0)[:P]
                    p_fire = 1.0 - (1.0 - probs) ** k_cap
                    fire_th = float(os.environ.get("OW_V3_FIRE_TH", "0.5"))
                    above = (p_fire >= fire_th).nonzero(as_tuple=False).flatten()
                    if int(above.numel()) == 0:
                        continue                             # hold
                    order_t = above[torch.argsort(p_fire[above],
                                                  descending=True)]
                else:  # "selfthresh" — the original hard-threshold decode
                    above = (row > self_logit).nonzero(as_tuple=False).flatten()
                    if int(above.numel()) == 0:
                        continue                             # self wins: hold
                    order_t = above[torch.argsort(row[above], descending=True)]
                fired = order_t[:k_cap].tolist()
                rem = n_ships - int(min_launch) * len(fired)
                alloc_logits = torch.cat([
                    pair_frac_raw[s, fired],
                    pair_frac_raw[s, s].reshape(1),
                ])
                share = torch.softmax(alloc_logits, dim=-1)
                # v4 stochastic allocation: draw shares from the learned
                # Dirichlet (α = α0[s] · mean) instead of using the mean.
                # OW_V4_ALLOC=dirichlet; requires the α0 head (v4 ckpts).
                if (os.environ.get("OW_V4_ALLOC") == "dirichlet"
                        and "alloc_conc" in preds):
                    a0 = preds["alloc_conc"].squeeze(0)[s].clamp(min=1e-3)
                    share = torch.distributions.Dirichlet(
                        (a0 * share).clamp(min=1e-4)).sample()
                if getattr(self, "_shape_debug", False):
                    cat_l = torch.cat([
                        row, torch.tensor([self_logit], device=row.device)])
                    p_all = torch.softmax(cat_l, dim=0)
                    srt = torch.sort(share[:-1], descending=True).values
                    self.__dict__.setdefault("_shape_rows", []).append({
                        "step": step, "n_legal": int((row > -1e30).sum()),
                        "p_self_sel": float(p_all[-1]),
                        "sel_top1": float(p_all[:P].max()),
                        "n_fired": len(fired),
                        "hold_share": float(share[-1]),
                        "alloc_top1": float(srt[0]) if len(srt) else float("nan"),
                        "alloc_ppl": float(torch.exp(-(share * (share + 1e-9).log()).sum())),
                    })
                for j, t in enumerate(fired):
                    ships_t = int(min_launch) + float(share[j]) * max(0, rem)
                    picked.append((float(row[t]), s, int(t),
                                   ships_t / max(1, n_ships)))
            if not picked:
                return []
            picked.sort(key=lambda x: -x[0])                 # logit-descending
            actions = []
            for _logit, s, t, frac in picked:
                source_pid = idx_to_pid.get(s)
                target_pid = idx_to_pid.get(t)
                if source_pid is None or target_pid is None:
                    continue
                actions.append((source_pid, target_pid, float(frac)))
            return actions
        elif inference_mode == "alloc_softmax":
            # bernoulli_select_multinomial_alloc_v2 deploy rule. Fired set =
            # the threshold rule's (every legal cell whose SELECT logit clears
            # ``logit_threshold``); sizes = the contract's allocation softmax
            # ``share = softmax([frac_loc[s, fired], frac_loc[s, s]])`` so the
            # HOLD share (frac diagonal) keeps ships home. Each cell's share is
            # later scaled by source.ships in ``act()`` — same pipeline as
            # threshold mode, whose surplus budget still caps total spend.
            firing = (pair_logits > logit_threshold) & cell_legal
            if not firing.any():
                return []
            src_indices, tgt_indices = firing.nonzero(as_tuple=True)
        else:  # threshold
            # Per-cell: every legal cell whose logit clears the threshold.
            firing = (pair_logits > logit_threshold) & cell_legal
            if not firing.any():
                return []
            src_indices, tgt_indices = firing.nonzero(as_tuple=True)

        alloc_share = None
        if inference_mode == "alloc_softmax":
            # Row-wise masked softmax over [fired cells ... , HOLD diagonal].
            # Launchable rows always carry their HOLD slot, so no row in
            # ``src_indices`` can be all--inf (no NaNs reach the lookup).
            diag = torch.arange(P, device=device)
            alloc_mask = firing.clone()
            alloc_mask[diag, diag] |= src_legal
            alloc_logits = pair_frac_raw.masked_fill(~alloc_mask, neg_inf)
            alloc_share = torch.softmax(alloc_logits, dim=-1)

        # Convert to a list of (src, tgt, frac). To keep the per-source
        # launches well-ordered when surplus is tight, emit cells in
        # pair_logits-descending order (matters for threshold mode where
        # one source may have many firing cells competing for its ships).
        cell_logits = pair_logits[src_indices, tgt_indices]
        order = torch.argsort(cell_logits, descending=True)
        actions: list[tuple[int, int, float]] = []
        for k in order.tolist():
            src_idx = int(src_indices[k].item())
            tgt_idx = int(tgt_indices[k].item())
            source_pid = idx_to_pid.get(src_idx)
            target_pid = idx_to_pid.get(tgt_idx)
            if source_pid is None or target_pid is None:
                continue
            if alloc_share is not None:
                frac = float(alloc_share[src_idx, tgt_idx].item())
            else:
                frac = float(torch.sigmoid(pair_frac_raw[src_idx, tgt_idx]).item())
            actions.append((source_pid, target_pid, frac))
        return actions

    @torch.no_grad()
    def value_forward(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Featurize ``obs`` and return the model's full prediction dict
        (incl. value heads when present). Used by the K-rank deploy agent
        to score simulated post-action states; mirrors act()'s
        featurize → L0 → forward pipeline without the decode."""
        learner_slot = int(obs.get("player_id", 0) if isinstance(obs, dict)
                           else getattr(obs, "player_id", 0))
        batch, _pid_to_idx = featurize_observation(
            obs,
            learner_slot=learner_slot,
            tracker=FleetTracker(),
            num_players=self.num_players,
            max_planets=self.max_planets,
            max_fleets=self.max_fleets,
            device=self.device,
        )
        B, P, _ = batch["planet_features"].shape
        comet_features = torch.zeros(
            (B, P, self.comet_enc.input_dim), device=self.device,
            dtype=batch["planet_features"].dtype,
        )
        comet_features[..., :18] = batch["planet_features"][..., :18]
        is_comet = batch["planet_features"][..., 0] > 0.5
        planet_tok = self.planet_enc(batch["planet_features"])
        comet_tok = self.comet_enc(comet_features)
        fleet_tok = self.fleet_enc(batch["fleet_features"])
        entity_self = _build_entity_self_tokens(planet_tok, comet_tok, is_comet)
        routing = {
            "fleet_target_idx": batch["fleet_target_idx"],
            "fleet_source_idx": batch["fleet_source_idx"],
            "fleet_owner_slot": batch["fleet_owner_slot"],
            "fleet_ships_log": batch["fleet_ships_log"],
            "fleet_eta_norm": batch["fleet_eta_norm"],
            "fleet_mask": batch["fleet_mask"],
        }
        from .pretrain.entity_encoder import (
            ENTITY_N_OWNER_CLASSES as _N_OWN,
            _PLANET_OWNER_START_IDX as _OWN0,
        )
        return self.model.forward_with_context(
            entity_self, fleet_tok, routing, batch["planet_mask"],
            is_comet=is_comet,
            pair_type_ids=build_pair_type_ids(
                batch["planet_features"], batch["planet_mask"],
            ),
            planet_owner_oh=batch["planet_features"][..., _OWN0:_OWN0 + _N_OWN],
        )

    def act(self, obs: dict[str, Any]) -> list[list]:
        """Return ``[[planet_id, angle, ships], ...]`` action triples.

        Sources can emit multiple launches per turn (one per target whose
        ``pair_logits[s, t]`` cleared the threshold). To prevent the same
        source's ship count from being double-spent across its targets,
        we budget the source's ``compute_surplus`` once and consume from
        a per-source running remainder as we walk the actions in
        logit-descending order.
        """
        learner_slot = int(obs.get("player", 0)) if isinstance(obs, dict) else int(obs.player)
        actions = self._predict(obs, learner_slot)
        if not actions:
            return []

        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        step = int(get("step") or 0)
        _neutral_bonus, defense_buffer, min_launch, _safety, _frontier_w, _eta_tol = (
            PHASE_TABLE[phase_of(step)]
        )
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        enemy_fleets = [f for f in fleets if f.owner != learner_slot and f.owner >= 0]
        by_id = {p.id: p for p in planets}

        # Per-source budget: starts at compute_surplus(source, ...), drains
        # as launches commit. We also cache source.ships (used to scale
        # frac into a ship count) so multi-target rows share the same
        # baseline rather than re-reading mutated obs.
        per_source_budget: dict[int, int] = {}
        per_source_total_ships: dict[int, int] = {}
        for source_pid, _t, _f in actions:
            if source_pid in per_source_budget:
                continue
            src = by_id.get(int(source_pid))
            if src is None:
                continue
            # Default = LIFTED (user call: model freedom over heuristics) —
            # the learned HOLD share alone decides what stays home for v3
            # decodes. The 5-update A/B had measured guard 11/20 vs lift 8/20,
            # but that was with a barely-calibrated alloc head; as it sharpens
            # the lift's premise strengthens. OW_V3_TRUST_GARRISON=0 restores
            # the compute_surplus guard for comparison runs.
            if self.inference_mode == "topk_self" and os.environ.get(
                    "OW_V3_TRUST_GARRISON", "1") == "1":
                # v3 garrison-trust deploy: the surplus reserve is LIFTED —
                # the learned HOLD share (frac diagonal) decides what stays
                # home; the env allows full evacuation and sampled training
                # never had a reserve. OW_V3_TRUST_GARRISON=0 restores the
                # compute_surplus guard (A/B arm — the greedy decode re-fires
                # EVERY step, so an uncapped budget can evacuate continuously
                # in a way the sampled contract never does). Legacy decodes
                # always keep the guard.
                per_source_budget[source_pid] = int(src.ships)
            else:
                per_source_budget[source_pid] = int(compute_surplus(
                    src, enemy_fleets, defense_buffer,
                ))
            per_source_total_ships[source_pid] = int(src.ships)

        moves: list[list] = []
        for source_pid, target_pid, frac in actions:
            budget = per_source_budget.get(source_pid, 0)
            if budget < min_launch:
                continue
            base_ships = per_source_total_ships.get(source_pid, 0)
            if self.inference_mode in ("alloc_softmax", "topk_self"):
                # Contract-faithful sizing: counts ARE share·ships (v2) or
                # floor+extras (v3); cells whose allocation lands below
                # min_launch are DROPPED, not floored up (tiny spillover
                # should not be inflated into launches).
                desired = int(round(frac * base_ships))
                if desired < min_launch:
                    continue
            else:
                desired = max(min_launch, int(round(frac * base_ships)))
            ships = min(budget, desired)
            if ships < min_launch:
                continue
            emitted = self._target_to_moves(
                source_pid, target_pid, obs, learner_slot,
                ships_override=ships,
            )
            if emitted:
                # _target_to_moves may have clamped via its fallback path;
                # subtract the actually-emitted ship count from the budget.
                actually_spent = int(emitted[0][2])
                per_source_budget[source_pid] = max(0, budget - actually_spent)
                moves.extend(emitted)
        return moves

    @staticmethod
    def _target_to_moves(
        source_pid: int,
        target_pid: int,
        obs: dict[str, Any],
        learner_slot: int,
        frac_override: float | None = None,
        ships_override: int | None = None,
    ) -> list[list]:
        """Translate (source, target) into a single ``[planet_id, angle,
        ships]`` triple.

        ``ships_override`` (preferred when set by the caller) pins the
        exact ship count — callers managing a per-source surplus budget
        across multiple targets use this to avoid double-spending the
        same source's ships. ``frac_override`` is the legacy single-launch
        path: caller passes a fraction and the function multiplies it by
        ``source.ships`` before clamping to surplus.

        Learned fractions are kept only when ``plan_launch`` validates
        the exact source→target trajectory. If that fixed-size launch is
        invalid, fall back to physical_v4's validated candidate sizing
        for the same target.
        """
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        raw_planets = get("planets") or []
        raw_fleets = get("fleets") or []
        initial_planets = get("initial_planets") or []
        angular_velocity = abs(float(get("angular_velocity") or 0.0))
        step = int(get("step") or 0)
        comet_planet_ids = set(get("comet_planet_ids") or [])
        comets = get("comets") or []

        phase = phase_of(step)
        neutral_bonus, defense_buffer, min_launch, safety, frontier_w, _eta_tol = (
            PHASE_TABLE[phase]
        )
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        av_signed = (
            infer_rotation_sign(planets, initial_planets) * angular_velocity
        )
        raw_by_id = {row[0]: row for row in raw_planets}

        source = next((p for p in planets if p.id == source_pid), None)
        target = next((p for p in planets if p.id == target_pid), None)
        if source is None or target is None or source.owner != learner_slot:
            return []
        my_planets = [p for p in planets if p.owner == learner_slot]
        enemy_fleets = [
            f for f in fleets if f.owner != learner_slot and f.owner >= 0
        ]
        surplus = compute_surplus(source, enemy_fleets, defense_buffer)

        # Path 1: caller pinned the ship count (multi-target ``act()`` budgeter).
        # No fallback — if the trajectory is invalid at this size, drop the
        # launch rather than reshape it.
        if ships_override is not None:
            ships = min(int(ships_override), int(surplus))
            if ships < min_launch:
                return []
            planned = _validated_learner_launch(
                source_raw=raw_by_id.get(source.id),
                target_raw=raw_by_id.get(target.id),
                ships=ships,
                planets=raw_planets,
                fleets=raw_fleets,
                initial_planets=initial_planets,
                angular_velocity=angular_velocity,
                comet_ids=comet_planet_ids,
                comets=comets,
                player=learner_slot,
                surplus=surplus,
                safety_buffer=safety,
                current_step=step,
            )
            if planned is None:
                return []
            angle, _eta = planned
            return [[source.id, angle, ships]]

        # Path 2: legacy frac-only path. Validate learned-size first; fall
        # back to physical_v4 candidate sizing only when validation fails.
        if frac_override is not None and surplus >= min_launch:
            ships = min(
                int(surplus),
                max(min_launch, int(round(frac_override * source.ships))),
            )
            if ships < min_launch:
                return []
            planned = _validated_learner_launch(
                source_raw=raw_by_id.get(source.id),
                target_raw=raw_by_id.get(target.id),
                ships=ships,
                planets=raw_planets,
                fleets=raw_fleets,
                initial_planets=initial_planets,
                angular_velocity=angular_velocity,
                comet_ids=comet_planet_ids,
                comets=comets,
                player=learner_slot,
                surplus=surplus,
                safety_buffer=safety,
                current_step=step,
            )
            if planned is not None:
                angle, _eta = planned
                return [[source.id, angle, ships]]
        if surplus < min_launch:
            return []

        # Fallback: physical_v4 surplus-based sizing when frac override
        # is unavailable or yields too few ships.
        cands = candidates_for_source(
            source,
            [target],
            my_planets,
            surplus,
            av_signed,
            angular_velocity,
            neutral_bonus,
            frontier_w,
            safety,
            raw_planets,
            raw_fleets,
            raw_by_id,
            learner_slot,
            step,
            comet_planet_ids=comet_planet_ids,
            comets=comets,
        )
        if not cands:
            return []
        _t, _eta, ships, angle, _score = cands[0]
        return [[source.id, angle, ships]]


# ---- Registry hook ----
_AGENT_SINGLETON: TransformerAgent | None = None


def transformer_v2_agent(obs):
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = TransformerAgent.load()
    return _AGENT_SINGLETON.act(obs)


# Frozen baseline checkpoint. Pinned to the best d=256 / 8-head /
# 5-head-PairHead run trained on the bow+Ebi cache; row R@1 = 0.4165,
# row R@5 = 0.7445, pair_logits BCE = 0.0506 on the held-out split.
# When a fresh run lands under ``data/runs/entity/<new-TS>/`` the
# default ckpt resolver picks it up (newest mtime wins), but this
# constant + the ``transformer_v2_baseline`` registration below keep
# the prior agent loadable for head-to-head matches.
BASELINE_CKPT = (
    _REPO_ROOT / "data" / "runs" / "entity"
    / "bowEbi_pair5head_d256_h8_lr5e-05_b128_30ep_20260520-095412"
    / "entity_encoder_best.pt"
)

_BASELINE_SINGLETON: TransformerAgent | None = None


def transformer_v2_baseline_agent(obs):
    """Frozen-baseline counterpart of ``transformer_v2``.

    Always loads from :data:`BASELINE_CKPT` (the May-20 8-head ckpt)
    regardless of what other runs land under ``data/runs/entity/``.
    Use this to A/B the live ``transformer_v2`` against a known-good
    baseline after a retrain — e.g.::

        python run.py --mode play --agents transformer_v2 transformer_v2_baseline

    The constant lives in :mod:`agents.transformer_v2.runner` so the
    same loader path (and therefore the same legacy-ckpt compat shim)
    is used for both ids.
    """
    global _BASELINE_SINGLETON
    if _BASELINE_SINGLETON is None:
        if not BASELINE_CKPT.exists():
            raise FileNotFoundError(
                f"transformer_v2_baseline ckpt missing at {BASELINE_CKPT}. "
                "Pull it from "
                "gs://orbit-wars-shipping/entity/runs/"
                "bowEbi_pair5head_d256_h8_lr5e-05_b128_30ep_20260520-095412/"
                "entity_encoder_best.pt"
            )
        _BASELINE_SINGLETON = TransformerAgent.load(ckpt_path=BASELINE_CKPT)
    return _BASELINE_SINGLETON.act(obs)


# Idempotent registration: ``python -m agents.transformer_v2.runner``
# imports this module twice (once via the package import chain, once
# as ``__main__``). A bare ``@register`` would raise on the second pass.
from .. import registry as _registry  # noqa: E402

if "transformer_v2" not in _registry._REGISTRY:
    _registry.register(
        "transformer_v2",
        "v2 entity-pretrain PairHead policy: L0 specialists + L1-L4 + 2-head "
        "FiLM PairHead, thresholded pair-score cells + pair_frac sizing.",
    )(transformer_v2_agent)

if "transformer_v2_baseline" not in _registry._REGISTRY:
    _registry.register(
        "transformer_v2_baseline",
        "Frozen May-20 8-head d=256 5-head-PairHead baseline (bow+Ebi cache, "
        "epoch 29). Use for A/B against the live transformer_v2 after a "
        "retrain. ckpt: bowEbi_pair5head_d256_h8_lr5e-05_b128_30ep_20260520-095412.",
    )(transformer_v2_baseline_agent)


# ---- Inference-variant agents -----------------------------------------
# Same loaded ckpt as ``transformer_v2`` (the newest under data/runs/entity/),
# but each variant uses a different decision rule on top of pair_logits.
# Useful in the dashboard for A/B-ing thresholds + flat / per-source argmax
# fallbacks without retraining.
#
# Each variant carries its own singleton (so per-episode FleetTracker state
# stays clean across A/B matches) at a modest memory cost (~14 MB ckpt
# weights + ~few MB activations per slot).

_VARIANT_REGISTRY: dict[str, TransformerAgent | None] = {}


def _make_variant_agent(name: str, *, inference_mode: str, logit_threshold: float):
    """Build a registered agent callable that lazily loads the latest
    ckpt with the requested inference rule."""
    def _fn(obs):
        agent = _VARIANT_REGISTRY.get(name)
        if agent is None:
            agent = TransformerAgent.load(
                inference_mode=inference_mode,
                logit_threshold=logit_threshold,
            )
            _VARIANT_REGISTRY[name] = agent
        return agent.act(obs)
    _fn.__name__ = f"{name}_agent"
    _fn.__doc__ = (
        f"transformer_v2 + inference_mode={inference_mode!r}"
        + (f", logit_threshold={logit_threshold}" if inference_mode == "threshold" else "")
    )
    return _fn


_VARIANT_SPECS: tuple[tuple[str, str, float, str], ...] = (
    (
        "transformer_v2_thr1",
        "threshold", 1.0,
        "transformer_v2 with logit threshold=1.0 (aggressive — fires more pair cells; "
        "diagnose under-launching on the threshold=2.0 default).",
    ),
    (
        "transformer_v2_thr3",
        "threshold", 3.0,
        "transformer_v2 with logit threshold=3.0 (conservative — only highly-confident "
        "cells fire; diagnose over-launching / wasted fleets).",
    ),
    (
        "transformer_v2_rowargmax",
        "row_argmax", 0.0,
        "transformer_v2 with per-source argmax: each launchable source picks its "
        "single best target, no threshold. At most one launch per source per turn.",
    ),
    (
        "transformer_v2_flatargmax",
        "flat_argmax", 0.0,
        "transformer_v2 with flat-grid argmax: single best (src, tgt) over the full "
        "P×P grid. At most one launch per turn — the original v2 single-shot rule.",
    ),
)

for _name, _mode, _thr, _desc in _VARIANT_SPECS:
    if _name not in _registry._REGISTRY:
        _registry.register(_name, _desc)(
            _make_variant_agent(_name, inference_mode=_mode, logit_threshold=_thr),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Load the agent and verify a single act() call.")
    args = parser.parse_args()
    if args.smoke_test:
        agent = TransformerAgent.load(args.ckpt, device=args.device)
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": 1729})
        obs = env.steps[0][0].observation
        moves = agent.act(obs)
        print(f"smoke-test moves: {moves}")


if __name__ == "__main__":
    main()
