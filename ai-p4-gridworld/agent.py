import math
import numpy as np
import os
import pickle
from typing import Optional

MOVES = ["N", "E", "S", "W"]

# Directional deltas as (dx, dy)
MOVE_DELTAS = {
    "N": (0,  1),   # y increases going north  (y=39 is north wall)
    "S": (0, -1),   # y decreases going south  (y=0  is south wall)
    "E": (1,  0),
    "W": (-1, 0),
}

GOAL_REWARD_THRESHOLD = 1000.0
TRAP_REWARD_THRESHOLD = -500.0


class GridNavigator:
    """
    Two-phase Q-learning agent for GridWorld.

    Phase 1 — Discovery: explores with decaying per-cell epsilon until
               the goal is found for the first time.
    Phase 2 — Exploitation: epsilon is locked near-zero so the agent
               follows learned Q-values (which encode the actual
               navigable path) to reach the goal every episode.
    """

    def __init__(self, world_id: str,
                 alpha: float = 0.2,
                 gamma: float = 0.95,
                 eps_start: float = 1.0,
                 eps_floor: float = 0.15,
                 eps_floor_exploit: float = 0.01,
                 eps_decay: float = 0.995):

        self.world_id = world_id
        self._q_path    = f"qtables/q_world_{world_id}.npy"
        self._meta_path = f"qtables/meta_{world_id}.pkl"

        self.alpha = alpha
        self.gamma = gamma
        self._eps_start         = eps_start
        self._eps_floor         = eps_floor           # floor during discovery
        self._eps_floor_exploit = eps_floor_exploit   # floor once goal is known
        self._eps_decay         = eps_decay

        self._is_continuation = os.path.exists(self._q_path)
        self._q  = self._load_q()
        meta     = self._load_meta()

        # Per-cell epsilon grid (same x/y dims as q-table)
        self._cell_eps: np.ndarray = meta.get("cell_eps", None)
        if self._cell_eps is None:
            self._cell_eps = np.full((40, 40), eps_start)

        self.danger_cells: set             = meta.get("danger_cells", set())
        self.goal_cell:    Optional[tuple] = meta.get("goal_cell", None)

        # Visit counts for UCB exploration bonus
        self._visit_counts: np.ndarray = meta.get("visit_counts", None)
        if self._visit_counts is None:
            self._visit_counts = np.zeros((40, 40), dtype=np.int32)
        self._total_visits: int = int(self._visit_counts.sum())

        # Persisted session counters (total_steps, episode, etc.)
        self.session_stats: dict = meta.get("session_stats", {})

        # Switch to exploit phase immediately if goal already known
        if self.goal_cell is not None:
            self._lock_exploit_mode()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def goal_found(self) -> bool:
        return self.goal_cell is not None

    @property
    def mean_epsilon(self) -> float:
        return float(self._cell_eps.mean())

    @property
    def is_continuation(self) -> bool:
        return self._is_continuation

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def pick_action(self, pos: tuple) -> tuple:
        """
        Returns (action_index, phase_label).

        Phase 1 (discovery): epsilon-greedy with safe random moves.
        Phase 2 (exploitation): near-greedy Q-value follow.
        """
        self._grow_if_needed(pos)
        x, y = pos
        eps = self._cell_eps[x][y]

        if np.random.rand() < eps:
            return self._safe_random(pos), "discover"

        return self._greedy_action(pos), "exploit"

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def record_transition(self, pos: tuple, action: int,
                          reward: float, next_pos: tuple):
        """Standard Q-learning update + per-cell epsilon decay."""
        self._grow_if_needed(pos)
        self._grow_if_needed(next_pos)
        x, y   = pos
        nx, ny = next_pos

        best_next = float(np.max(self._q[nx][ny]))
        self._q[x][y][action] += self.alpha * (
            reward + self.gamma * best_next - self._q[x][y][action]
        )
        self._visit_counts[x][y] += 1
        self._total_visits += 1
        self._decay_cell(pos)

    def record_terminal(self, pos: tuple, action: int, reward: float):
        """Q-update for a terminal transition (self-loop, no next state)."""
        self.record_transition(pos, action, reward, pos)

    # ------------------------------------------------------------------
    # Goal / danger marking
    # ------------------------------------------------------------------

    def flag_goal(self, pos: tuple):
        self.goal_cell = tuple(pos)
        print(f"[Phase 2] Goal locked at {pos} — switching to exploitation mode.")
        self._lock_exploit_mode()

    def flag_danger(self, pos: tuple):
        self.danger_cells.add(tuple(pos))
        print(f"[Danger] Cell {pos} marked hazardous. "
              f"Total known: {len(self.danger_cells)}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, _path=None):
        os.makedirs("qtables", exist_ok=True)
        np.save(self._q_path, self._q)
        with open(self._meta_path, "wb") as f:
            pickle.dump({
                "cell_eps":      self._cell_eps,
                "danger_cells":  self.danger_cells,
                "goal_cell":     self.goal_cell,
                "visit_counts":  self._visit_counts,
                "session_stats": self.session_stats,
            }, f)

    def load(self, _path=None):
        self._q = self._load_q()
        meta = self._load_meta()
        self._cell_eps     = meta.get("cell_eps", np.full((40, 40), self._eps_start))
        self.danger_cells  = meta.get("danger_cells", set())
        self.goal_cell     = meta.get("goal_cell", None)
        self._visit_counts = meta.get("visit_counts", np.zeros((40, 40), dtype=np.int32))
        self._total_visits = int(self._visit_counts.sum())
        self.session_stats = meta.get("session_stats", {})
        if self.goal_cell is not None:
            self._lock_exploit_mode()
        print(f"[Load] world={self.world_id} | "
              f"goal={self.goal_cell} | "
              f"dangers={len(self.danger_cells)} | "
              f"mean_eps={self.mean_epsilon:.3f} | "
              f"total_visits={self._total_visits}")

    # ------------------------------------------------------------------
    # Compatibility shims (keep main.py / train.py working unchanged)
    # ------------------------------------------------------------------

    @property
    def q_table(self):
        return self._q

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

    @property
    def epsilon(self):
        return self.mean_epsilon

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_q(self) -> np.ndarray:
        if os.path.exists(self._q_path):
            return np.load(self._q_path)
        legacy_path = f"qtables/q_{self.world_id}.npy"
        if os.path.exists(legacy_path):
            return np.load(legacy_path)
        return np.zeros((40, 40, 4))

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
        new_x = max(max_x, x + 1)
        new_y = max(max_y, y + 1)

        grown_q = np.zeros((new_x, new_y, 4))
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
        x, y = pos
        floor = self._eps_floor_exploit if self.goal_found else self._eps_floor
        self._cell_eps[x][y] = max(
            floor, self._cell_eps[x][y] * self._eps_decay
        )

    def _lock_exploit_mode(self):
        """Clamp all cell epsilons down to the exploitation floor."""
        self._cell_eps = np.minimum(self._cell_eps, self._eps_floor_exploit)

    def _safe_neighbours(self, pos: tuple) -> list:
        """Return action indices that stay in-bounds and avoid danger cells."""
        x, y = pos
        max_x, max_y, _ = self._q.shape
        safe = []
        for idx, mv in enumerate(MOVES):
            dx, dy = MOVE_DELTAS[mv]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < max_x and 0 <= ny < max_y):
                continue
            if (nx, ny) in self.danger_cells:
                continue
            safe.append(idx)
        return safe

    def _safe_random(self, pos: tuple) -> int:
        """Random action biased toward less-visited neighbours."""
        safe = self._safe_neighbours(pos)
        if not safe:
            return int(np.random.randint(4))
        x, y = pos
        max_x, max_y, _ = self._q.shape
        weights = []
        for a in safe:
            dx, dy = MOVE_DELTAS[MOVES[a]]
            nx = min(max(x + dx, 0), max_x - 1)
            ny = min(max(y + dy, 0), max_y - 1)
            weights.append(1.0 / (1 + self._visit_counts[nx][ny]))
        w = np.array(weights, dtype=float)
        w /= w.sum()
        return int(np.random.choice(safe, p=w))

    def _greedy_action(self, pos: tuple) -> int:
        """Greedy action; uses UCB exploration bonus during discovery phase."""
        x, y = pos
        safe = self._safe_neighbours(pos)
        if not safe:
            return int(np.argmax(self._q[x][y]))
        if not self.goal_found:
            # UCB bonus drives the agent toward unvisited cells
            log_t = math.log1p(self._total_visits)
            max_x, max_y, _ = self._q.shape
            def ucb_score(a):
                dx, dy = MOVE_DELTAS[MOVES[a]]
                nx = min(max(x + dx, 0), max_x - 1)
                ny = min(max(y + dy, 0), max_y - 1)
                bonus = 2.0 * math.sqrt(log_t / (1 + self._visit_counts[nx][ny]))
                return self._q[x][y][a] + bonus
            return max(safe, key=ucb_score)
        return max(safe, key=lambda a: self._q[x][y][a])


# ---------------------------------------------------------------------------
# Module-level alias so `from agent import QLearningAgent` still works
# ---------------------------------------------------------------------------
QLearningAgent = GridNavigator