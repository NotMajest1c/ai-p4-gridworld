import numpy as np
import os

ACTIONS = ["N", "E", "S", "W"]

class QLearningAgent:
    def __init__(self, world_id):
        self.world_id = world_id
        self.q_path = f"qtables/q_world_{world_id}.npy"

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

        self.q_table = self.load_q()

    def load_q(self):
        if os.path.exists(self.q_path):
            return np.load(self.q_path)
        return np.zeros((40, 40, 4))

    def _ensure_state_capacity(self, state):
        x, y = state
        max_x, max_y, _ = self.q_table.shape
        grow_x = x >= max_x
        grow_y = y >= max_y
        if not (grow_x or grow_y):
            return

        new_x = max(max_x, x + 1)
        new_y = max(max_y, y + 1)
        expanded = np.zeros((new_x, new_y, 4))
        expanded[:max_x, :max_y, :] = self.q_table
        self.q_table = expanded

    def save_q(self):
        os.makedirs("qtables", exist_ok=True)
        np.save(self.q_path, self.q_table)

    def choose_action(self, state):
        action_idx, _ = self.choose_action_with_mode(state)
        return action_idx

    def choose_action_with_mode(self, state):
        self._ensure_state_capacity(state)
        x, y = state
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(4)), "explore"
        return int(np.argmax(self.q_table[x][y])), "exploit"

    def update(self, state, action, reward, next_state):
        self._ensure_state_capacity(state)
        self._ensure_state_capacity(next_state)

        x, y = state
        nx, ny = next_state

        best_next = np.max(self.q_table[nx][ny])

        self.q_table[x][y][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[x][y][action]
        )