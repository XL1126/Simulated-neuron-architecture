"""
Grid-based virtual world v2 with walls, corridors, rooms, and collectible objects.

Provides a much richer environment than EnhancedVirtualWorld, with:
  - Grid-based world with walls forming rooms and corridors
  - Object types: food (+1), poison (-1), key (+0.5), treasure (+2)
  - Exploration bonus for visiting new cells
  - Meaningful sensory vectors encoding real spatial information
  - Discrete action space with collision handling

  Observation space (sensory_package dict):
    visual     (20 dims) – 8-direction raycast: object type + distance per direction
                             last 4 dims: global remaining object counts
    auditory   (14 dims) – nearest distance + direction to each of 4 object types
                             last 2 dims: exploration ratio, danger ratio
    tactile    (12 dims) – 8 adjacent cells (N/NE/E/SE/S/SW/W/NW): wall/object/empty
                             last 4 dims: collision flag, heading, normalized pos
    vestibular (10 dims) – facing direction cos/sin, last action, step phase,
                             exploration ratio, cumulative reward, oscillation signals
    place_cells(20 dims) – 4×5 grid of Gaussian receptive fields over normalized pos
    olfactory  ( 8 dims) – scent intensity: food, poison, treasure, novelty,
                             net valence, time, keys remaining, recent-reward flag

  Action space:
    4 discrete actions decoded from motor_output[0] % 4:
      0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

  Reward signal:
    food        +1.0
    poison      -1.0
    key         +0.5
    treasure    +2.0
    exploration +0.05  (first visit to a cell)
    wall bump   -0.05
    step cost   -0.001
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Grid-world environment
# ---------------------------------------------------------------------------

class GridWorldV2:
    """Grid-based virtual world with structured rooms, corridors, and objects.

    Public interface matches EnhancedVirtualWorld so it can be dropped into
    ExperimentRunner via the ``world_class`` parameter.
    """

    # ── constructor ────────────────────────────────────────────────────

    def __init__(self, size: float = 20.0, seed: int = 42):
        """Create a grid world.

        Parameters
        ----------
        size : float
            World size; converted to ``grid_size = max(10, int(size))``.
        seed : int
            Random seed for reproducibility.
        """
        self.grid_size = max(10, int(size))
        self.rng = random.Random(seed)
        np_random = np.random.RandomState(seed)

        # Public attributes (used by runner)
        self.size = size                     # kept for compatibility
        self.step_count = 0
        self.success_count = 0
        self.total_reward = 0.0              # cumulative reward this episode

        # Agent state
        self.agent_pos = np.array(
            [self.grid_size // 2, self.grid_size // 2], dtype=np.float32
        )
        self.agent_dir = 0                   # 0=up, 1=right, 2=down, 3=left
        self.last_action = 0
        self.last_reward = 0.0

        # Grid layers
        #   grid[y, x] == 0 → walkable    1 → wall
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        #   object_grid[y, x] == -1 → empty,  else → object_id
        self.object_grid = np.full(
            (self.grid_size, self.grid_size), -1, dtype=np.int32
        )
        self.objects: Dict[int, dict] = {}   # id → {type, pos, collected}
        self._next_object_id = 0

        # Exploration tracking
        self.visited = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int32
        )

        # Build world
        self._generate_walls(np_random)
        self._spawn_objects(np_random)
        self._mark_visited()

    # ── world generation ───────────────────────────────────────────────

    def _generate_walls(self, np_random: np.random.RandomState):
        """Create a structured layout with rooms and corridors."""
        gs = self.grid_size
        # Outer boundary
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Horizontal room dividers at thirds
        for row in [gs // 3, 2 * gs // 3]:
            self.grid[row, 2:gs - 2] = 1
            # corridor gaps
            gaps = sorted(np_random.choice(
                range(2, gs - 2),
                size=np_random.randint(2, 4),
                replace=False,
            ))
            for c in gaps:
                self.grid[row, c] = 0

        # Vertical dividers and pillars
        for col in [gs // 4, 3 * gs // 4]:
            # partial walls with gaps
            seg = np_random.choice([0, 1], size=gs - 4, p=[0.35, 0.65])
            for i, v in enumerate(seg):
                r = i + 2
                if r not in (gs // 3, 2 * gs // 3):  # don't block corridor crossings
                    self.grid[r, col] = v

        # A few extra pillars / L-walls for variety
        for _ in range(np_random.randint(4, 8)):
            r = np_random.randint(2, gs - 2)
            c = np_random.randint(2, gs - 2)
            if self.grid[r, c] == 0 and self.grid[r, c] != 1:
                self.grid[r, c] = 1

        # Ensure start cell and its neighbours are walkable
        sy, sx = int(self.agent_pos[1]), int(self.agent_pos[0])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < gs and 0 <= nx < gs:
                    self.grid[ny, nx] = 0

    def _is_walkable(self, y: int, x: int) -> bool:
        return (0 <= y < self.grid_size and 0 <= x < self.grid_size
                and self.grid[y, x] == 0)

    def _random_empty_cell(self) -> Optional[Tuple[int, int]]:
        candidates = [
            (y, x)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if self.grid[y, x] == 0 and self.object_grid[y, x] == -1
        ]
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _spawn_objects(self, np_random: np.random.RandomState):
        """Place objects on empty walkable cells."""
        # Food  – 8 pieces
        self._place_objects("food", 8)
        # Poison – 6 pieces
        self._place_objects("poison", 6)
        # Key    – 3 pieces
        self._place_objects("key", 3)
        # Treasure – 2 pieces
        self._place_objects("treasure", 2)

    def _place_objects(self, obj_type: str, count: int):
        for _ in range(count):
            cell = self._random_empty_cell()
            if cell is None:
                break
            y, x = cell
            oid = self._next_object_id
            self._next_object_id += 1
            self.object_grid[y, x] = oid
            self.objects[oid] = {
                "type": obj_type,
                "pos": np.array([x, y], dtype=np.float32),
                "collected": False,
            }

    def _mark_visited(self):
        y, x = int(self.agent_pos[1]), int(self.agent_pos[0])
        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
            self.visited[y, x] = 1

    # ── queries ────────────────────────────────────────────────────────

    def exploration_ratio(self) -> float:
        """Fraction of walkable cells that have been visited."""
        walkable = int(np.sum(self.grid == 0))
        visited = int(np.sum((self.visited == 1) & (self.grid == 0)))
        return visited / max(1, walkable)

    def object_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"food": 0, "poison": 0, "key": 0, "treasure": 0}
        for obj in self.objects.values():
            if not obj["collected"]:
                counts[obj["type"]] = counts.get(obj["type"], 0) + 1
        return counts

    # ── sensory package (matches EnhancedVirtualWorld interface) ───────

    def get_sensory_package(self) -> Dict[str, Any]:
        """Return dict with visual/auditory/tactile/vestibular/place_cells/olfactory.

        Every vector element encodes real spatial or temporal information —
        nothing is left as random noise.
        """
        visual = np.zeros(20, dtype=np.float32)
        auditory = np.zeros(14, dtype=np.float32)
        tactile = np.zeros(12, dtype=np.float32)
        vestibular = np.zeros(10, dtype=np.float32)
        place_cells = np.zeros(20, dtype=np.float32)
        olfactory = np.zeros(8, dtype=np.float32)

        ay, ax = int(self.agent_pos[1]), int(self.agent_pos[0])
        gs = self.grid_size

        # ── Visual: 8-direction raycast ──────────────────────────────
        # 2 indices per direction (16 total); remaining 4 = global counts
        _DIRS = [
            (-1,  0),  # N
            (-1,  1),  # NE
            (0,   1),  # E
            (1,   1),  # SE
            (1,   0),  # S
            (1,  -1),  # SW
            (0,  -1),  # W
            (-1, -1),  # NW
        ]

        for di, (dy, dx) in enumerate(_DIRS):
            i0 = di * 2
            i1 = i0 + 1
            for dist in range(1, 8):
                ny, nx = ay + dy * dist, ax + dx * dist
                if not (0 <= ny < gs and 0 <= nx < gs):
                    visual[i0] = max(visual[i0], -0.1)  # out-of-bounds signal
                    break
                if self.grid[ny, nx] == 1:               # wall
                    visual[i0] = max(visual[i0], 1.0 - dist * 0.12)
                    break
                oid = self.object_grid[ny, nx]
                if oid >= 0:
                    obj = self.objects.get(oid)
                    if obj and not obj["collected"]:
                        w = 1.0 - dist * 0.15
                        if obj["type"] == "food":
                            visual[i0] = max(visual[i0], w * 0.8)
                            visual[i1] = max(visual[i1], w * 0.4)
                        elif obj["type"] == "poison":
                            visual[i0] = max(visual[i0], w * 0.6)
                            visual[i1] = max(visual[i1], -w * 0.4)
                        elif obj["type"] == "key":
                            visual[i1] = max(visual[i1], w * 0.7)
                        elif obj["type"] == "treasure":
                            visual[i0] = max(visual[i0], w * 0.9)
                            visual[i1] = max(visual[i1], w * 0.9)

        # Global object-count indicators
        cnt = self.object_counts()
        visual[16] = min(1.0, cnt["food"] / 8.0)
        visual[17] = min(1.0, cnt["poison"] / 6.0)
        visual[18] = min(1.0, cnt["key"] / 3.0)
        visual[19] = min(1.0, cnt["treasure"] / 2.0)

        # ── Auditory: nearest-object distance + direction per type ───
        # 4 types × 3 dims = 12  +  2 global signals = 14
        for ti, otype in enumerate(["food", "poison", "key", "treasure"]):
            b = ti * 3
            best_dist = float("inf")
            best_dy, best_dx = 0.0, 0.0
            for obj in self.objects.values():
                if obj["collected"] or obj["type"] != otype:
                    continue
                d = float(np.linalg.norm(obj["pos"] - self.agent_pos))
                if d < best_dist:
                    best_dist = d
                    best_dy = float(obj["pos"][1] - self.agent_pos[1])
                    best_dx = float(obj["pos"][0] - self.agent_pos[0])
            if best_dist < float("inf"):
                auditory[b] = max(0.0, 1.0 - best_dist / gs)
                auditory[b + 1] = float(np.clip(best_dy / max(0.01, best_dist), -1, 1))
                auditory[b + 2] = float(np.clip(best_dx / max(0.01, best_dist), -1, 1))

        auditory[12] = self.exploration_ratio()
        total_danger = cnt["poison"] + cnt["food"] + cnt["treasure"]
        auditory[13] = cnt["poison"] / max(1, total_danger)

        # ── Tactile: 8 adjacent cells + 4 meta signals ───────────────
        adjacent = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1),
        ]
        for i, (dy, dx) in enumerate(adjacent):
            ny, nx = ay + dy, ax + dx
            if not (0 <= ny < gs and 0 <= nx < gs):
                tactile[i] = -0.3          # boundary
            elif self.grid[ny, nx] == 1:
                tactile[i] = -0.6          # wall
            else:
                oid = self.object_grid[ny, nx]
                if oid >= 0:
                    obj = self.objects.get(oid)
                    if obj and not obj["collected"]:
                        tmap = {"food": 0.5, "poison": -0.9,
                                "key": 0.35, "treasure": 0.8}
                        tactile[i] = tmap.get(obj["type"], 0.15)
                    else:
                        tactile[i] = 0.05
                else:
                    tactile[i] = 0.05       # empty walkable

        tactile[8] = 1.0 if self.last_reward < -0.01 else 0.0
        tactile[9] = self.agent_dir / 4.0
        tactile[10] = float(ay) / gs
        tactile[11] = float(ax) / gs

        # ── Vestibular: orientation + movement + temporal signals ────
        rad = self.agent_dir * np.pi / 2.0
        vestibular[0] = float(np.cos(rad))
        vestibular[1] = float(np.sin(rad))
        vestibular[2] = self.last_action / 4.0
        vestibular[3] = (self.step_count % 100) / 100.0
        vestibular[4] = self.exploration_ratio()
        vestibular[5] = float(np.clip(self.total_reward / 10.0, -1, 1))
        vestibular[6] = float(np.sin(self.agent_pos[0] * 0.5))
        vestibular[7] = float(np.cos(self.agent_pos[1] * 0.5))
        vestibular[8] = float(np.sin(self.step_count * 0.1))
        vestibular[9] = float(np.cos(self.step_count * 0.13))

        # ── Place cells: 4×5 Gaussian receptive fields ───────────────
        cx = float(self.agent_pos[0]) / gs
        cy = float(self.agent_pos[1]) / gs
        for i in range(20):
            cell_cx = (i % 5 + 0.5) / 5.0
            cell_cy = (i // 5 + 0.5) / 4.0
            d = np.sqrt((cx - cell_cx) ** 2 + (cy - cell_cy) ** 2)
            place_cells[i] = float(np.exp(-d * 5.0))

        # ── Olfactory: scent gradients ────────────────────────────────
        food_scent = 0.0
        poison_scent = 0.0
        treasure_scent = 0.0
        for obj in self.objects.values():
            if obj["collected"]:
                continue
            d = float(np.linalg.norm(obj["pos"] - self.agent_pos))
            if d < 5.0:
                s = 1.0 - d / 5.0
                if obj["type"] == "food":
                    food_scent = max(food_scent, s)
                elif obj["type"] == "poison":
                    poison_scent = max(poison_scent, s)
                elif obj["type"] == "treasure":
                    treasure_scent = max(treasure_scent, s)

        olfactory[0] = food_scent
        olfactory[1] = poison_scent
        olfactory[2] = treasure_scent
        olfactory[3] = self.exploration_ratio()
        olfactory[4] = (food_scent - poison_scent) * 0.5 + 0.5
        olfactory[5] = float(self.step_count % 200) / 200.0
        olfactory[6] = cnt["key"] / 3.0
        olfactory[7] = 1.0 if self.last_reward > 0.01 else 0.0

        return {
            "visual": visual.tolist(),
            "auditory": auditory.tolist(),
            "tactile": tactile.tolist(),
            "vestibular": vestibular.tolist(),
            "place_cells": place_cells.tolist(),
            "olfactory": olfactory.tolist(),
        }

    # ── text description ───────────────────────────────────────────────

    def describe(self) -> str:
        """Natural-language description of agent's immediate surroundings."""
        ay, ax = int(self.agent_pos[1]), int(self.agent_pos[0])

        tokens: List[str] = []
        dir_labels = {(-1, 0): "north", (0, 1): "east",
                      (1, 0): "south", (0, -1): "west"}
        for (dy, dx), label in dir_labels.items():
            ny, nx = ay + dy, ax + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.grid[ny, nx] == 1:
                    tokens.append(f"wall_{label}")
                else:
                    oid = self.object_grid[ny, nx]
                    if oid >= 0:
                        obj = self.objects.get(oid)
                        if obj and not obj["collected"]:
                            tokens.append(f"{obj['type']}_{label}")

        # Nearby objects within radius 3
        for obj in self.objects.values():
            if obj["collected"]:
                continue
            d = float(np.linalg.norm(obj["pos"] - self.agent_pos))
            if d < 4.0:
                dy = float(obj["pos"][1] - self.agent_pos[1])
                dx = float(obj["pos"][0] - self.agent_pos[0])
                if abs(dy) > abs(dx):
                    dirstr = "south" if dy > 0 else "north"
                else:
                    dirstr = "east" if dx > 0 else "west"
                tokens.append(f"{obj['type']}_{dirstr}")

        return " ".join(tokens[:12]) if tokens else "empty"

    # ── step + interaction ─────────────────────────────────────────────

    def tick(self):
        """Advance world state: respawn collected objects periodically."""
        self.step_count += 1

        # Respawn some collected objects after a while
        if self.step_count % 150 == 0:
            for oid, obj in list(self.objects.items()):
                if obj["collected"] and self.rng.random() < 0.25:
                    cell = self._random_empty_cell()
                    if cell is not None:
                        y, x = cell
                        obj["collected"] = False
                        obj["pos"] = np.array([x, y], dtype=np.float32)
                        self.object_grid[y, x] = oid

    def try_interact(self, motor_output) -> float:
        """Apply an action decoded from *motor_output* and return reward.

        ``motor_output[0] % 4`` selects a discrete direction:
        0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
        """
        if motor_output is None or len(motor_output) == 0:
            return 0.0

        action = int(abs(float(motor_output[0]))) % 4
        self.last_action = action

        ay, ax = int(self.agent_pos[1]), int(self.agent_pos[0])

        # Direction mapping
        _MOVE = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dy, dx = _MOVE[action]
        ny, nx = ay + dy, ax + dx

        reward = 0.0
        self.last_reward = 0.0

        # Small step cost – encourages efficiency
        reward -= 0.001

        # Wall collision
        if not self._is_walkable(ny, nx):
            reward -= 0.05
            self.last_reward = reward
            self.total_reward += reward
            return reward

        # Move agent
        self.agent_pos = np.array([nx, ny], dtype=np.float32)
        self.agent_dir = action

        # Exploration bonus
        was_visited = self.visited[ny, nx]
        self._mark_visited()
        if not was_visited:
            reward += 0.05

        # Object interaction
        oid = self.object_grid[ny, nx]
        if oid >= 0:
            obj = self.objects.get(oid)
            if obj and not obj["collected"]:
                obj["collected"] = True
                self.object_grid[ny, nx] = -1

                rmap = {"food": 1.0, "poison": -1.0, "key": 0.5, "treasure": 2.0}
                r = rmap.get(obj["type"], 0.0)
                reward += r
                self.success_count += (2 if obj["type"] == "treasure" else 1)

        self.last_reward = reward
        self.total_reward += reward
        return reward