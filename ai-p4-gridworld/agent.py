"""
GridNavigator — Enhanced Q-Learning Agent
==========================================
Improvements over original:

1.  Wall detection        — stall moves → mark wall, penalise Q, never repeat
2.  Known-edge graph      — every successful move recorded for BFS
3.  BFS path-following    — once goal known, navigate optimally, not by chance
4.  Manhattan fallback    — if BFS path not yet available, move toward goal by Manhattan distance
5.  Value propagation     — after EACH goal arrival, backward-BFS boots Q-values
6.  Prioritised replay    — replay high-TD-error past transitions (32× learning)
7.  Adaptive alpha        — higher LR on rarely-visited cells → faster early learn
8.  Optimistic Q-init     — small +0.01 bias drives full exploration naturally
9.  gamma=0.99            — long paths (60-80 steps) need high gamma or reward vanishes
"""

import math
import numpy as np
import os
import pickle
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

MOVES = ["N", "E", "S", "W"]

MOVE_DELTAS: Dict[str, Tuple[int, int]] = {
    "N": (0,  1),
    "S": (0, -1),
    "E": (1,  0),
    "W": (-1, 0),
}

GOAL_REWARD_THRESHOLD = 1000.0
TRAP_REWARD_THRESHOLD = -500.0


class GridNavigator:
    """
    Two-phase Q-learning agent:
      Phase 1 (Discover)  — epsilon-greedy + UCB bonus, builds wall/edge map.
      Phase 2 (Exploit)   — BFS on known edges for shortest path to goal.
    """

    def __init__(
        self,
        world_id: str,
        alpha: float = 0.25,          # base learning rate (adaptive on top)
        gamma: float = 0.99,          # high gamma needed: paths can be 60-80 steps
        eps_start: float = 1.0,
        eps_floor: float = 0.10,
        eps_floor_exploit: float = 0.02,
        eps_decay: float = 0.99,      # faster decay than original 0.995
        replay_capacity: int = 10_000,
        replay_batch: int = 32,
    ):
        self.world_id = world_id
        self._q_path    = f"qtables/q_world_{world_id}.npy"
        self._meta_path = f"qtables/meta_{world_id}.pkl"

        self.alpha              = alpha
        self.gamma              = gamma
        self._eps_start         = eps_start
        self._eps_floor         = eps_floor
        self._eps_floor_exploit = eps_floor_exploit
        self._eps_decay         = eps_decay

        # ── Experience replay ─────────────────────────────────────────
        self._replay_buf: deque = deque(maxlen=replay_capacity)
        self._replay_batch      = replay_batch

        # ── Load state ────────────────────────────────────────────────
        #self._is_continuation = False  
        #self._q = np.zeros_like(self._load_q())     
        #meta = {}
        self._is_continuation = os.path.exists(self._q_path)
        self._q               = self._load_q()
        meta                  = self._load_meta()

        self._cell_eps: np.ndarray = meta.get(
            "cell_eps", np.full((40, 40), eps_start)
        )
        self.danger_cells: Set[Tuple]    = meta.get("danger_cells", set())
        self.goal_cell: Optional[Tuple]  = meta.get("goal_cell", None)

        self._visit_counts: np.ndarray = meta.get(
            "visit_counts", np.zeros((40, 40), dtype=np.int32)
        )
        self._total_visits: int = int(self._visit_counts.sum())

        # ── Wall map: pos → set of blocked action indices ─────────────
        self.wall_map: Dict[Tuple, Set[int]] = meta.get("wall_map", {})

        # ── Known navigable edges: pos → {action_idx: next_pos} ──────
        self._known_edges: Dict[Tuple, Dict[int, Tuple]] = meta.get(
            "known_edges", {}
        )

        # BFS path cache (list of action indices remaining)
        self._bfs_cache: Optional[List[int]] = None
        self._bfs_start: Optional[Tuple]     = None

        self.session_stats: dict = meta.get("session_stats", {})

        if self.goal_cell is not None:
            self._lock_exploit_mode()

    # ──────────────────────────────────────────────────────────────────
    # Public properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def goal_found(self) -> bool:
        return self.goal_cell is not None

    @property
    def mean_epsilon(self) -> float:
        return float(self._cell_eps.mean())

    @property
    def is_continuation(self) -> bool:
        return self._is_continuation

    @property
    def q_table(self) -> np.ndarray:
        return self._q

    # ──────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────

    def pick_action(self, pos: tuple) -> Tuple[int, str]:
        """
        Returns (action_index, phase_label).

        Exploit phase: follow BFS path if available, else greedy Q.
        Discover phase: epsilon-greedy with UCB exploration bonus.
        """
        self._grow_if_needed(pos)
        x, y = pos

        # ── BFS path following (goal known) ───────────────────────────
        if self.goal_found:
            action = self._next_bfs_action(pos)
            if action is not None:
                return action, "exploit[BFS]"
            # BFS has no connected path yet → fall back to Manhattan distance
            action = self._manhattan_action(pos)
            if action is not None:
                return action, "exploit[Manhattan]"
            # Manhattan also blocked → fall through to greedy

        eps = float(self._cell_eps[x][y])
        if np.random.rand() < eps:
            return self._random_action(pos), "discover"
        return self._greedy_action(pos), "exploit"

    # ──────────────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────────────

    def record_transition(
        self, pos: tuple, action: int, reward: float, next_pos: tuple
    ):
        """Q-update for a non-terminal step, then replay."""
        self._grow_if_needed(pos)
        self._grow_if_needed(next_pos)
        x, y   = pos
        nx, ny = next_pos

        # Detect wall (agent didn't move)
        if pos == next_pos:
            self._mark_wall(pos, action)
        else:
            self._record_edge(pos, action, next_pos)

        # Adaptive alpha: faster learning on lightly-visited cells
        visits = int(self._visit_counts[x][y])
        alpha  = min(0.60, self.alpha * (1.0 + 1.0 / (1 + visits)))

        best_next = float(np.max(self._q[nx][ny]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self._q[x][y][action]
        self._q[x][y][action] += alpha * td_error

        self._visit_counts[x][y] += 1
        self._total_visits       += 1
        self._decay_cell(pos)

        # Store (is_terminal=False)
        self._replay_buf.append((pos, action, reward, next_pos, abs(td_error), False))
        self._replay()

    def record_terminal(self, pos: tuple, action: int, reward: float):
        """Q-update for a terminal transition (goal / trap / episode-end)."""
        self._grow_if_needed(pos)
        x, y = pos
        visits = int(self._visit_counts[x][y])
        alpha  = min(0.60, self.alpha * (1.0 + 1.0 / (1 + visits)))

        td_error = reward - self._q[x][y][action]  # no bootstrap at terminal
        self._q[x][y][action] += alpha * td_error

        self._visit_counts[x][y] += 1
        self._total_visits       += 1
        self._decay_cell(pos)

        self._replay_buf.append((pos, action, reward, pos, abs(td_error), True))
        self._replay()

    # ──────────────────────────────────────────────────────────────────
    # Goal / danger marking
    # ──────────────────────────────────────────────────────────────────

    def flag_goal(self, pos: tuple):
        self.goal_cell = tuple(pos)
        print(f"[Phase 2] Goal locked at {pos} — BFS mode active.")
        self._lock_exploit_mode()
        self._propagate_goal_value(pos)   # bootstrap Q-values instantly
        self._bfs_cache = None            # force fresh path computation

    def flag_danger(self, pos: tuple):
        self.danger_cells.add(tuple(pos))
        print(f"[Danger] {pos} marked. Total danger: {len(self.danger_cells)}")

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save(self, _path=None):
        os.makedirs("qtables", exist_ok=True)
        np.save(self._q_path, self._q)
        with open(self._meta_path, "wb") as f:
            pickle.dump(
                {
                    "cell_eps":      self._cell_eps,
                    "danger_cells":  self.danger_cells,
                    "goal_cell":     self.goal_cell,
                    "visit_counts":  self._visit_counts,
                    "wall_map":      self.wall_map,
                    "known_edges":   self._known_edges,
                    "session_stats": self.session_stats,
                },
                f,
            )

    def load(self, _path=None):
        self._q            = self._load_q()
        meta               = self._load_meta()
        self._cell_eps     = meta.get("cell_eps",     np.full((40, 40), self._eps_start))
        self.danger_cells  = meta.get("danger_cells", set())
        self.goal_cell     = meta.get("goal_cell",    None)
        self._visit_counts = meta.get("visit_counts", np.zeros((40, 40), dtype=np.int32))
        self._total_visits = int(self._visit_counts.sum())
        self.wall_map      = meta.get("wall_map",     {})
        self._known_edges  = meta.get("known_edges",  {})
        self.session_stats = meta.get("session_stats",{})
        self._bfs_cache    = None
        if self.goal_cell is not None:
            self._lock_exploit_mode()
        edges_count = sum(len(v) for v in self._known_edges.values())
        walls_count = sum(len(v) for v in self.wall_map.values())
        print(
            f"[Load] world={self.world_id} | goal={self.goal_cell} | "
            f"dangers={len(self.danger_cells)} | "
            f"known_edges={edges_count} | known_walls={walls_count} | "
            f"mean_eps={self.mean_epsilon:.3f} | visits={self._total_visits}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Compatibility shims (keep main.py unchanged)
    # ──────────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self.mean_epsilon

    def choose_action_with_mode(self, pos):
        return self.pick_action(pos)

    def update(self, pos, action, reward, next_pos):
        self.record_transition(pos, action, reward, next_pos)

    def mark_goal(self, pos):
        self.flag_goal(pos)

    def mark_trap(self, pos):
        self.flag_danger(pos)

    def state_to_idx(self, s: dict) -> tuple:
        return int(s["x"]), int(s["y"])

    def idx_to_state(self, idx) -> dict:
        x, y = idx
        return {"x": x, "y": y}

    # ──────────────────────────────────────────────────────────────────
    # Internal: Q-table management
    # ──────────────────────────────────────────────────────────────────

    def _load_q(self) -> np.ndarray:
        if os.path.exists(self._q_path):
            q = np.load(self._q_path)
            # Keep existing values; only set 0-cells to optimistic init
            q[q == 0.0] = 0.01
            return q
        legacy = f"qtables/q_{self.world_id}.npy"
        if os.path.exists(legacy):
            return np.load(legacy)
        # Optimistic initialisation: small positive bias drives exploration
        return np.full((40, 40, 4), 0.01, dtype=float)

    def _load_meta(self) -> dict:
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _grow_if_needed(self, pos: tuple):
        x, y = pos
        max_x, max_y, _ = self._q.shape
        if x < max_x and y < max_y:
            return
        new_x, new_y = max(max_x, x + 1), max(max_y, y + 1)

        grown_q = np.full((new_x, new_y, 4), 0.01)
        grown_q[:max_x, :max_y, :] = self._q
        self._q = grown_q

        grown_eps = np.full((new_x, new_y), self._eps_start)
        ex, ey = self._cell_eps.shape
        grown_eps[:ex, :ey] = self._cell_eps
        self._cell_eps = grown_eps

        grown_vc = np.zeros((new_x, new_y), dtype=np.int32)
        vx, vy = self._visit_counts.shape
        grown_vc[:vx, :vy] = self._visit_counts
        self._visit_counts = grown_vc

    def _decay_cell(self, pos: tuple):
        x, y  = pos
        floor = self._eps_floor_exploit if self.goal_found else self._eps_floor
        self._cell_eps[x][y] = max(floor, self._cell_eps[x][y] * self._eps_decay)

    def _lock_exploit_mode(self):
        self._cell_eps = np.minimum(self._cell_eps, self._eps_floor_exploit)

    # ──────────────────────────────────────────────────────────────────
    # Internal: Wall & edge tracking
    # ──────────────────────────────────────────────────────────────────

    def _mark_wall(self, pos: tuple, action: int):
        """Record confirmed wall; penalise that Q entry."""
        pos = tuple(pos)
        if pos not in self.wall_map:
            self.wall_map[pos] = set()
        if action not in self.wall_map[pos]:
            self.wall_map[pos].add(action)
            x, y = pos
            # Strong negative to prevent re-trying the wall
            self._q[x][y][action] = min(self._q[x][y][action], -1.0)
            print(f"  [Wall detected] {pos} → {MOVES[action]} is blocked")

    def _record_edge(self, pos: tuple, action: int, next_pos: tuple):
        """Record navigable transition; invalidate BFS cache if new."""
        pos, next_pos = tuple(pos), tuple(next_pos)
        if pos not in self._known_edges:
            self._known_edges[pos] = {}
        if self._known_edges[pos].get(action) != next_pos:
            self._known_edges[pos][action] = next_pos
            self._bfs_cache = None  # graph changed

    # ──────────────────────────────────────────────────────────────────
    # Internal: Experience replay
    # ──────────────────────────────────────────────────────────────────

    def _replay(self):
        """Sample a prioritised mini-batch and do off-policy Q-updates."""
        buf = self._replay_buf
        if len(buf) < max(self._replay_batch, 64):
            return

        items  = list(buf)
        errors = np.array([t[4] for t in items], dtype=float) + 1e-6
        probs  = errors / errors.sum()

        idxs = np.random.choice(len(items), size=self._replay_batch, replace=False, p=probs)
        for i in idxs:
            pos, action, reward, next_pos, _, is_terminal = items[i]
            x,  y  = pos
            nx, ny = next_pos
            self._grow_if_needed(pos)
            self._grow_if_needed(next_pos)

            if is_terminal:
                target = reward
            else:
                target = reward + self.gamma * float(np.max(self._q[nx][ny]))

            self._q[x][y][action] += self.alpha * (target - self._q[x][y][action])

    # ──────────────────────────────────────────────────────────────────
    # Internal: BFS path planning
    # ──────────────────────────────────────────────────────────────────

    def _next_bfs_action(self, pos: tuple) -> Optional[int]:
        """Return next action from cached BFS path; recompute if stale."""
        # If cache is valid and head matches current pos
        if self._bfs_cache is not None and self._bfs_start == pos and self._bfs_cache:
            action = self._bfs_cache[0]
            # Sanity: ensure action not a known wall
            if action not in self.wall_map.get(tuple(pos), set()):
                self._bfs_cache = self._bfs_cache[1:]
                # Update expected next position for next call
                dx, dy = MOVE_DELTAS[MOVES[action]]
                x, y   = pos
                self._bfs_start = (x + dx, y + dy)
                return action
            else:
                # Cache stale (wall found after caching)
                self._bfs_cache = None

        # (Re-)compute path
        path = self._bfs_to_goal(pos)
        if path:
            action          = path[0]
            self._bfs_cache = path[1:]
            dx, dy          = MOVE_DELTAS[MOVES[action]]
            x, y            = pos
            self._bfs_start = (x + dx, y + dy)
            return action
        return None  # no known path yet

    def _bfs_to_goal(self, start: tuple) -> Optional[List[int]]:
        """Shortest path in known_edges graph from start to goal_cell."""
        if self.goal_cell is None:
            return None
        goal = self.goal_cell

        if start == goal:
            return []

        queue:   deque = deque([(start, [])])
        visited: Set[Tuple] = {start}

        while queue:
            pos, path = queue.popleft()
            blocked   = self.wall_map.get(tuple(pos), set())
            for a_idx, nxt in self._known_edges.get(pos, {}).items():
                if a_idx in blocked:
                    continue
                if nxt in self.danger_cells:
                    continue
                if nxt in visited:
                    continue
                new_path = path + [a_idx]
                if nxt == goal:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return None  # goal not reachable through known edges yet

    # ──────────────────────────────────────────────────────────────────
    # Internal: Value propagation after goal discovery
    # ──────────────────────────────────────────────────────────────────

    def _propagate_goal_value(self, goal_pos: tuple):
        """
        Backward BFS from goal through known_edges.
        Instantly sets Q[prev][action] ≈ γ^d * GOAL_REWARD for every cell
        reachable in d steps from goal.  This dramatically speeds up
        subsequent exploitation episodes.
        """
        if not self._known_edges:
            return

        # Build reverse graph
        reverse: Dict[Tuple, List[Tuple]] = {}
        for pos, edges in self._known_edges.items():
            for a_idx, nxt in edges.items():
                reverse.setdefault(nxt, []).append((pos, a_idx))

        queue: deque = deque([(goal_pos, 0)])
        visited: Set[Tuple] = {goal_pos}

        count = 0
        while queue:
            pos, depth = queue.popleft()
            # Value discounted by distance from goal
            value = GOAL_REWARD_THRESHOLD * (self.gamma ** depth)

            for prev_pos, a_idx in reverse.get(pos, []):
                px, py = prev_pos
                self._grow_if_needed(prev_pos)
                boosted = value * 0.95  # slight discount so live rewards still matter
                if self._q[px][py][a_idx] < boosted:
                    self._q[px][py][a_idx] = boosted
                    count += 1
                if prev_pos not in visited:
                    visited.add(prev_pos)
                    queue.append((prev_pos, depth + 1))

        print(
            f"[Value Propagation] Bootstrapped {count} Q-entries "
            f"across {len(visited)} cells."
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal: Action helpers
    # ──────────────────────────────────────────────────────────────────

    def _safe_actions(self, pos: tuple) -> List[int]:
        """Actions that avoid known walls, grid edges, and danger cells."""
        x, y             = pos
        max_x, max_y, _  = self._q.shape
        blocked          = self.wall_map.get(tuple(pos), set())
        safe             = []
        for idx, mv in enumerate(MOVES):
            if idx in blocked:
                continue
            dx, dy = MOVE_DELTAS[mv]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < max_x and 0 <= ny < max_y):
                continue
            if (nx, ny) in self.danger_cells:
                continue
            safe.append(idx)
        return safe

    def _random_action(self, pos: tuple) -> int:
        """Inverse-visit-count weighted random action from safe set."""
        safe = self._safe_actions(pos)
        if not safe:
            return int(np.random.randint(4))
        x, y             = pos
        max_x, max_y, _  = self._q.shape
        weights          = []
        for a in safe:
            dx, dy = MOVE_DELTAS[MOVES[a]]
            nx = min(max(x + dx, 0), max_x - 1)
            ny = min(max(y + dy, 0), max_y - 1)
            weights.append(1.0 / (1 + self._visit_counts[nx][ny]))
        w = np.array(weights, dtype=float)
        w /= w.sum()
        return int(np.random.choice(safe, p=w))

    def _manhattan_action(self, pos: tuple) -> Optional[int]:
        """
        Referans koddan götürülmüş fikir: BFS yol tapılmadıqda
        goal-a Manhattan məsafəsini minimuma endirən safe action seç.
        Qrafda boşluq olsa belə hər zaman bir cavab verir.
        """
        if self.goal_cell is None:
            return None
        gx, gy = self.goal_cell
        x,  y  = pos
        safe   = self._safe_actions(pos)
        if not safe:
            return None

        best_action   = None
        best_distance = float("inf")
        max_x, max_y, _ = self._q.shape

        for a in safe:
            dx, dy = MOVE_DELTAS[MOVES[a]]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < max_x and 0 <= ny < max_y):
                continue
            dist = abs(gx - nx) + abs(gy - ny)
            if dist < best_distance:
                best_distance = dist
                best_action   = a

        return best_action

    def _greedy_action(self, pos: tuple) -> int:
        """Greedy action with UCB bonus during discovery phase."""
        x, y = pos
        safe = self._safe_actions(pos)
        if not safe:
            return int(np.argmax(self._q[x][y]))
        if not self.goal_found:
            log_t            = math.log1p(self._total_visits)
            max_x, max_y, _  = self._q.shape

            def ucb(a: int) -> float:
                dx, dy = MOVE_DELTAS[MOVES[a]]
                nx = min(max(x + dx, 0), max_x - 1)
                ny = min(max(y + dy, 0), max_y - 1)
                bonus = 2.0 * math.sqrt(log_t / (1 + self._visit_counts[nx][ny]))
                return self._q[x][y][a] + bonus

            return max(safe, key=ucb)
        return max(safe, key=lambda a: self._q[x][y][a])


# ---------------------------------------------------------------------------
# Alias so `from agent import QLearningAgent` still works
# ---------------------------------------------------------------------------
QLearningAgent = GridNavigator
