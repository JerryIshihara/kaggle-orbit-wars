"""PPO training stub.

The PPO loop and the multi-head action decoder it trained were removed
during the encoder-freeze + single-pair-head smoke test redesign. The
goal of that redesign is to first validate whether the frozen encoder
representation can support direct expert ``(source, target)`` pair
prediction (see ``agents/transformer_v1/pretrain/pair_score.py``);
re-introducing PPO comes after the pair head shows signal.

The previous implementation lives in git history (commit ``f4056d3``
and earlier). This file remains so:

  * ``from agents.transformer_v1 import ppo`` still succeeds (the
    package import chain in ``__init__.py`` triggers it).
  * ``run.py``'s ``transformer-ppo`` stage fails with a clear, actionable
    error rather than an attribute error.
"""

from __future__ import annotations

from typing import Any


_DECODER_REMOVED_MSG = (
    "transformer_v1 PPO is currently disabled — the action decoder it "
    "drove was removed during the encoder-freeze + pair-head redesign. "
    "Train the new PairScoreHead first via "
    "`python -m agents.transformer_v1.pretrain.pair_score`; re-enable "
    "PPO once a runtime decoder is back."
)


def train_ppo(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_DECODER_REMOVED_MSG)


def main() -> None:
    raise NotImplementedError(_DECODER_REMOVED_MSG)


if __name__ == "__main__":
    main()
