"""CNN v1 agent — convolutional policy over a rasterized board.

Planned design (see chat history for discussion):
- Input: 15 × 50 × 50 tensor (player-relative channels).
- Backbone: shallow stack with dilated convs, RF ~33 cells.
- Action head: gather features at each owned planet, predict
  (launch_gate, target_cell, ship_fraction).
- Warm-start via BC on sniper, then PPO self-play.

Not yet implemented — no @register call so this agent won't appear in
the registry until the model is ready.
"""
