"""CNN v1 agent — convolutional policy over a player-relative 50×50 raster.

--------------------------------------------------------------------------
Input channels (shape: 15 × 50 × 50, cell = 2 world units)
--------------------------------------------------------------------------
| idx | channel                    | what it encodes                              |
|-----|----------------------------|----------------------------------------------|
|  0  | my_planet_ships            | log1p(ships)/log(5000) at my planets         |
|  1  | enemy_planet_ships         | ... at enemy planets                         |
|  2  | neutral_planet_ships       | ... at neutral planets                       |
|  3  | my_planet_production       | production/5 at my planets                   |
|  4  | enemy_planet_production    | ... at enemy planets                         |
|  5  | neutral_planet_production  | ... at neutral planets                       |
|  6  | my_fleet_ships             | log1p(ships)/log(5000) at my fleets          |
|  7  | enemy_fleet_ships          | ... at enemy fleets                          |
|  8  | my_fleet_sin               | sin(heading) at my fleets                    |
|  9  | my_fleet_cos               | cos(heading) at my fleets                    |
| 10  | enemy_fleet_sin            | sin(heading) at enemy fleets                 |
| 11  | enemy_fleet_cos            | cos(heading) at enemy fleets                 |
| 12  | orbit_mask                 | 1 on cells containing an orbiting planet     |
| 13  | comet_mask                 | 1 on cells containing a comet                |
| 14  | sun_mask                   | static disc at grid cell (25, 25), radius 5  |

Scalar side-channel (7 floats, concatenated after the backbone):
  step/500, log1p(my_ships)/log5k, log1p(enemy_ships)/log5k,
  my_planet_count/10, enemy_planet_count/10, angular_velocity,
  active_comet_count/20.

--------------------------------------------------------------------------
Backbone (receptive field ≈ 33 cells ≈ 66 world units)
--------------------------------------------------------------------------
  Conv(15 → 32, 3×3)                               ReLU  RF=3
  Conv(32 → 64, 3×3)                               ReLU  RF=5
  Conv(64 → 64, 3×3, dilation=2)                   ReLU  RF=9
  Conv(64 → 64, 3×3, dilation=4)                   ReLU  RF=17
  Conv(64 → 64, 3×3, dilation=8)                   ReLU  RF=33

Scalar MLP: 7 → 32 → 32 (ReLU).

--------------------------------------------------------------------------
Action head (per owned planet)
--------------------------------------------------------------------------
  bilinear-sample feature map at planet (x, y) → concat with scalar MLP
  output → MLP(96 → 96) → three heads:

    launch_logit      ∈ ℝ           (sigmoid → launch gate)
    target_logits     ∈ ℝ^(50×50)   (argmax → target cell)
    ship_fraction_lg  ∈ ℝ           (sigmoid → ship fraction of garrison)

The target cell → angle via atan2(ty − py, tx − px); ships =
round(fraction × garrison), clipped to [1, garrison].

Status: untrained. Default weights → near-random valid actions. When
WEIGHTS_PATH exists, it loads on first call.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

BOARD_SIZE = 100.0
GRID = 50
CELL = BOARD_SIZE / GRID
NUM_CHANNELS = 15
SCALAR_DIM = 7
D_MODEL = 64
MAX_STEPS = 500.0
SHIPS_LOG_MAX = math.log(5000.0)
LAUNCH_THRESHOLD = 0.5

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "cnn_v1.pt"


def _cell(v: float) -> int:
    return max(0, min(GRID - 1, int(v / CELL)))


def featurize(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    raw_planets = get("planets") or []
    raw_fleets = get("fleets") or []
    comet_ids = set(get("comet_planet_ids") or [])
    angular_velocity = float(get("angular_velocity") or 0.0)
    step = int(get("step", 0) or 0)

    ch = torch.zeros(NUM_CHANNELS, GRID, GRID, dtype=torch.float32)

    sy, sx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
    dist = ((sx - GRID / 2) ** 2 + (sy - GRID / 2) ** 2).sqrt()
    ch[14] = (dist <= 5).float()

    my_ships = 0
    enemy_ships = 0
    my_planet_count = 0
    enemy_planet_count = 0
    my_planets = []

    for p in raw_planets:
        pid, owner, x, y, radius, ships, prod = p
        cx, cy = _cell(x), _cell(y)
        log_ships = math.log1p(ships) / SHIPS_LOG_MAX
        prod_norm = prod / 5.0
        is_orbiting = angular_velocity > 0 and math.hypot(x - 50, y - 50) + radius < 50
        if owner == player:
            ch[0, cy, cx] += log_ships
            ch[3, cy, cx] += prod_norm
            my_ships += ships
            my_planet_count += 1
            my_planets.append((pid, x, y, ships))
        elif owner == -1:
            ch[2, cy, cx] += log_ships
            ch[5, cy, cx] += prod_norm
        else:
            ch[1, cy, cx] += log_ships
            ch[4, cy, cx] += prod_norm
            enemy_ships += ships
            enemy_planet_count += 1
        if is_orbiting:
            ch[12, cy, cx] = 1.0
        if pid in comet_ids:
            ch[13, cy, cx] = 1.0

    for f in raw_fleets:
        fid, owner, x, y, angle, from_pid, ships = f
        cx, cy = _cell(x), _cell(y)
        log_ships = math.log1p(ships) / SHIPS_LOG_MAX
        if owner == player:
            ch[6, cy, cx] += log_ships
            ch[8, cy, cx] = math.sin(angle)
            ch[9, cy, cx] = math.cos(angle)
            my_ships += ships
        elif owner != -1:
            ch[7, cy, cx] += log_ships
            ch[10, cy, cx] = math.sin(angle)
            ch[11, cy, cx] = math.cos(angle)
            enemy_ships += ships

    scalars = torch.tensor(
        [
            step / MAX_STEPS,
            math.log1p(my_ships) / SHIPS_LOG_MAX,
            math.log1p(enemy_ships) / SHIPS_LOG_MAX,
            my_planet_count / 10.0,
            enemy_planet_count / 10.0,
            angular_velocity,
            len(comet_ids) / 20.0,
        ],
        dtype=torch.float32,
    )
    return ch, scalars, my_planets


class CNNv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, D_MODEL, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_MODEL, D_MODEL, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_MODEL, D_MODEL, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_MODEL, D_MODEL, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.head_trunk = nn.Sequential(
            nn.Linear(D_MODEL + 32, 96),
            nn.ReLU(inplace=True),
        )
        self.launch_head = nn.Linear(96, 1)
        self.target_head = nn.Linear(96, GRID * GRID)
        self.ship_head = nn.Linear(96, 1)
        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL + 32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, channels, scalars):
        feat_map = self.conv(channels)
        scalar_feat = self.scalar_mlp(scalars)
        return feat_map, scalar_feat

    def value(self, feat_map, scalar_feat):
        pooled = feat_map.mean(dim=(2, 3))
        return self.value_head(torch.cat([pooled, scalar_feat], dim=-1)).squeeze(-1)

    def act(self, feat_map, scalar_feat, planet_coords):
        n = planet_coords.size(0)
        if n == 0:
            empty = torch.empty(0, dtype=torch.float32)
            return empty, torch.empty(0, GRID * GRID), empty

        grid = planet_coords.clone()
        grid[:, 0] = (grid[:, 0] / BOARD_SIZE) * 2 - 1
        grid[:, 1] = (grid[:, 1] / BOARD_SIZE) * 2 - 1
        grid = grid.view(1, n, 1, 2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=False)
        per_planet = sampled.squeeze(-1).squeeze(0).transpose(0, 1)

        s = scalar_feat.expand(n, -1)
        x = self.head_trunk(torch.cat([per_planet, s], dim=-1))
        return (
            self.launch_head(x).squeeze(-1),
            self.target_head(x),
            self.ship_head(x).squeeze(-1),
        )


_MODEL: CNNv1 | None = None


def _get_model() -> CNNv1:
    global _MODEL
    if _MODEL is None:
        m = CNNv1()
        m.eval()
        if WEIGHTS_PATH.exists():
            try:
                m.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
            except Exception as e:
                warnings.warn(f"cnn_v1: failed to load weights: {e}")
        _MODEL = m
    return _MODEL


@register(
    "cnn_v1",
    "Convolutional policy over a 50×50 player-relative raster — "
    "15 channels, dilated stack, per-planet action head. Untrained.",
)
def cnn_v1_agent(obs):
    channels, scalars, my_planets = featurize(obs)
    if not my_planets:
        return []

    model = _get_model()
    with torch.no_grad():
        feat_map, scalar_feat = model(channels.unsqueeze(0), scalars.unsqueeze(0))
        coords = torch.tensor([[x, y] for (_, x, y, _) in my_planets], dtype=torch.float32)
        launch, targets, ship_frac = model.act(feat_map, scalar_feat, coords)

    launch_p = torch.sigmoid(launch)
    ship_p = torch.sigmoid(ship_frac)
    target_cells = targets.argmax(dim=-1)

    moves = []
    for i, (pid, x, y, ships) in enumerate(my_planets):
        if launch_p[i].item() < LAUNCH_THRESHOLD:
            continue
        tc = int(target_cells[i].item())
        tcy, tcx = divmod(tc, GRID)
        tx = (tcx + 0.5) * CELL
        ty = (tcy + 0.5) * CELL
        n = max(1, min(int(ships), int(round(ship_p[i].item() * ships))))
        if n < 1:
            continue
        angle = math.atan2(ty - y, tx - x)
        moves.append([pid, angle, n])

    return moves
