"""Pretraining drivers for transformer_v1 encoders.

Module-per-encoder, named after the encoder it trains:

  * ``fleet_encoder``   — multi-task pretrain for ``encoder.fleet_encoder``
  * ``planet_encoder``  — multi-task pretrain for ``encoder.planet_encoder``
    (planet + comet, scalar branch + 1D-conv trajectory branch + gated
    fusion + trajectory decoder)

Each module is runnable as ``python -m agents.transformer_v1.pretrain.<name>``.
The encoders themselves live next door under ``encoder/``; this package
keeps the dataset / training-loop code separate so the nn.Modules stay
small and import-cheap.
"""
