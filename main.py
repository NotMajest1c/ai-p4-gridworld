"""
main.py — GridWorld Q-Learning Runner
======================================
New in this version:
  • Option 5 — Multi-World Scheduler: auto-cycles through all 10 worlds
    and tracks how many successful goal-reaching runs each world has,
    ensuring quorum (5 per world) as fast as possible.
  • BFS path logging in per-step output (shows when BFS is active).
  • Re-entry cooldown respected after terminal episodes.
"""

import os
import time
from api import enter_world, get_location, move, reset
from agent import GridNavigator, MOVES as ACTIONS, GOAL_REWARD_THRESHOLD, TRAP_REWARD_THRESHOLD

# How many seconds to wait between moves (API guideline: ≥ 2 s)
MOVE_DELAY   = 2
# How many seconds to wait before re-entering after a terminal episode
ENTRY_DELAY  = 3
# Quorum target per world
QUORUM       = 5
# All world IDs in the project
ALL_WORLDS   = [str(i) for i in range(10)]


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pos(s: dict) -> tuple:
    return int(s["x"]), int(s["y"])


def _location_info() -> tuple:
    loc = get_location()
    if loc.get("code") != "OK":
        print(loc)
        return None, None
    world = str(loc.get("world", ""))
    raw   = loc.get("state", "")
    if not raw:
        return world, None
    try:
        x_str, y_str = raw.split(":")
        return world, (int(x_str), int(y_str))
    except (ValueError, TypeError):
        print({"code": "FAIL", "message": f"Unexpected location payload: {loc}"})
        return world, None


def _enter_world(world_id) -> bool:
    wid = str(world_id)
    res = enter_world(wid)
    if res.get("code") == "OK":
        print(res)
        return True
    msg = str(res.get("message", ""))
    if "currently in world" in msg and wid in msg:
        print(f"Already in world {wid}. Continuing.")
        return True
    print(res)
    return False


def _await_valid_pos(target_world, retries: int = 6) -> tuple:
    for attempt in range(retries):
        w, pos = _location_info()
        if str(w) == str(target_world) and pos is not None:
            return pos
        print(f"[Retry {attempt+1}/{retries}] world={w}, pos={pos}")
        time.sleep(3)
    return None


def _write_vgrid(log_path, navigator, world_id, total_steps, episode,
                 is_exploit, summary_lines, step_buffer, is_continuation=False):
    q     = navigator.q_table
    rows  = min(q.shape[0], 40)
    cols  = min(q.shape[1], 40)
    label = "CONTINUED" if is_continuation else "NEW"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            f"=== World {world_id} | Step {total_steps} | Ep {episode} | "
            f"Phase: {'EXPLOIT' if is_exploit else 'DISCOVER'} | {label} ===\n"
        )
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\nV-VALUE GRID (max Q per cell)  [col=X, row=Y]:\n")
        f.write("     " + "".join(f"{c:8}" for c in range(cols)) + "\n")
        for r in range(rows):
            f.write(
                f"{r:4} " +
                "".join(f"{float(q[r][c].max()):8.2f}" for c in range(cols)) + "\n"
            )
        f.write("\n--- Recent Steps ---\n")
        for s in step_buffer:
            f.write(s + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Manual mode
# ─────────────────────────────────────────────────────────────────────────────

def manual_mode(world_id):
    if not _enter_world(world_id):
        return
    while True:
        direction = input("Move (N/E/S/W or Q to quit): ").upper()
        if direction == "Q":
            break
        if direction not in ACTIONS:
            print("Invalid — use N, E, S, W, or Q.")
            continue
        print(move(world_id, direction))


# ─────────────────────────────────────────────────────────────────────────────
# Auto mode — single world, returns number of successful (goal) episodes
# ─────────────────────────────────────────────────────────────────────────────

def auto_mode(world_id, max_steps: int = 0, max_goal_eps: int = 0):
    """
    Run Q-learning on one world.

    Args:
        world_id:     World to run in.
        max_steps:    Stop after this many steps (0 = unlimited).
        max_goal_eps: Stop after this many goal-reaching episodes (0 = unlimited).

    Returns:
        Number of goal-reaching episodes completed this session.
    """
    target = str(world_id)
    if not _enter_world(target):
        return 0

    navigator = GridNavigator(target)

    use_old = input("Continue from previous? (y/n): ").lower()

    if use_old == "y":
        navigator.load()
    

    log_dir  = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"world_{target}.txt")

    step_buffer: list = []
    STEP_BUFFER       = 60

    pos = _await_valid_pos(target)
    if pos is None:
        print("[Error] Could not get a valid starting position.")
        return 0

    is_cont = navigator.is_continuation
    sess    = navigator.session_stats

    if is_cont and sess:
        episode        = sess.get("episode", 1)
        total_steps    = sess.get("total_steps", 0)
        total_reward   = sess.get("total_reward", 0.0)
        discover_steps = sess.get("discover_steps", 0)
        exploit_steps  = sess.get("exploit_steps", 0)
        stall_steps    = sess.get("stall_steps", 0)
        seen_cells     = sess.get("seen_cells", set()) | {pos}
        action_tally   = sess.get("action_tally", {a: 0 for a in ACTIONS})
    else:
        episode        = 1
        total_steps    = 0
        total_reward   = 0.0
        discover_steps = 0
        exploit_steps  = 0
        stall_steps    = 0
        seen_cells     = {pos}
        action_tally   = {a: 0 for a in ACTIONS}

    goal_eps_this_session = 0
    step                  = 0
    LOG_EVERY             = 20
    recent_rewards: list  = []

    def _sync():
        navigator.session_stats.update({
            "episode":        episode,
            "total_steps":    total_steps,
            "total_reward":   total_reward,
            "discover_steps": discover_steps,
            "exploit_steps":  exploit_steps,
            "stall_steps":    stall_steps,
            "seen_cells":     seen_cells,
            "action_tally":   action_tally,
        })

    print(f"\n{'='*64}")
    print(f"  World {target}  |  "
          f"Phase: {'EXPLOIT + BFS' if navigator.goal_found else 'DISCOVER'}")
    print(f"  {'[CONTINUED]' if is_cont and sess else '[NEW RUN]'} "
          f"step={total_steps} | ep={episode}")
    if navigator.goal_found:
        print(f"  Goal known at {navigator.goal_cell} | "
              f"known_edges={sum(len(v) for v in navigator._known_edges.values())} | "
              f"walls={sum(len(v) for v in navigator.wall_map.values())}")
    print(f"{'='*64}\n")

    while True:
        # ── Stopping conditions ──────────────────────────────────────
        if max_steps > 0 and total_steps >= max_steps:
            print(f"[Stop] Reached max_steps={max_steps} for world {target}.")
            break
        if max_goal_eps > 0 and goal_eps_this_session >= max_goal_eps:
            print(f"[Stop] Reached max_goal_eps={max_goal_eps} for world {target}.")
            break

        step        += 1
        total_steps += 1

        action_idx, phase = navigator.pick_action(pos)
        direction         = ACTIONS[action_idx]
        action_tally[direction] += 1

        if phase == "discover":
            discover_steps += 1
        else:
            exploit_steps  += 1

        # Show BFS status in action label
        bfs_active = (
            navigator.goal_found and
            navigator._bfs_cache is not None and
            len(navigator._bfs_cache) >= 0
        )
        phase_label = f"{phase}{'[BFS]' if bfs_active else ''}"

        x, y  = pos
        old_q = float(navigator.q_table[x][y][action_idx])
        res   = move(target, direction)

        if res.get("code") != "OK":
            print(res)
            break

        reward       = float(res.get("reward", 0.0))
        total_reward += reward
        recent_rewards.append(reward)
        if len(recent_rewards) > LOG_EVERY:
            recent_rewards.pop(0)

        run_id        = res.get("runId", "?")
        new_state_raw = res.get("newState")

        # ── Terminal state ───────────────────────────────────────────
        if new_state_raw is None:
            navigator.record_terminal(pos, action_idx, reward)

            if reward >= GOAL_REWARD_THRESHOLD:
                print(f"\n🏆 [Ep {episode}] GOAL REACHED at {pos} | "
                      f"r={reward:.1f} | run={run_id} | "
                      f"BFS={'YES' if bfs_active else 'NO'}")
                navigator.flag_goal(pos)
                goal_eps_this_session += 1

            elif reward <= TRAP_REWARD_THRESHOLD:
                print(f"\n💀 [Ep {episode}] TRAP at {pos} | r={reward:.1f} | run={run_id}")
                navigator.flag_danger(pos)

            else:
                print(f"\n⚠️  [Ep {episode}] Episode ended at {pos} | "
                      f"r={reward:.1f} | run={run_id}")

            _sync()
            navigator.save()
            time.sleep(ENTRY_DELAY)

            if not _enter_world(target):
                print("Failed to re-enter world. Stopping.")
                break

            new_pos = _await_valid_pos(target)
            if new_pos is None:
                print("[Error] Could not get valid position after re-entry.")
                break

            episode += 1
            step     = 1
            # Reset BFS cache — new starting position
            navigator._bfs_cache = None

            print(f"\n--- Ep {episode} | "
                  f"Phase: {'EXPLOIT+BFS' if navigator.goal_found else 'DISCOVER'} | "
                  f"pos={new_pos} ---\n")

            pos = new_pos
            continue

        # ── Normal step ──────────────────────────────────────────────
        new_pos = _parse_pos(new_state_raw)
        if new_pos == pos:
            stall_steps += 1
        seen_cells.add(new_pos)

        nx, ny    = new_pos
        best_next = float(navigator.q_table[nx][ny].max())
        td_target = reward + navigator.gamma * best_next
        td_error  = td_target - old_q

        navigator.record_transition(pos, action_idx, reward, new_pos)
        new_q = float(navigator.q_table[x][y][action_idx])
        _sync()
        navigator.save()

        step_line = (
            f"Run {run_id} | Ep {episode} | Step {step} | "
            f"{pos} → {new_pos} | {direction}({phase_label}) | "
            f"r={reward:.3f} | dQ={td_error:+.4f} | newQ={new_q:.4f}"
        )
        print(step_line)
        step_buffer.append(step_line)
        if len(step_buffer) > STEP_BUFFER:
            step_buffer.pop(0)

        # ── Periodic summary ─────────────────────────────────────────
        if total_steps % LOG_EVERY == 0:
            avg_all    = total_reward / total_steps
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            stall_rate = stall_steps / total_steps

            ranked = sorted(
                (
                    (c, float(navigator.q_table[c[0]][c[1]].max()))
                    for c in seen_cells
                ),
                key=lambda t: t[1], reverse=True,
            )
            edges_count = sum(len(v) for v in navigator._known_edges.values())
            walls_count = sum(len(v) for v in navigator.wall_map.values())

            summary_lines = [
                f"Summary @{total_steps} steps | "
                f"avgR(all)={avg_all:.4f} | avgR(recent)={avg_recent:.4f} | "
                f"stall={stall_rate:.1%} | cells={len(seen_cells)}",
                f"Phase: {'EXPLOIT+BFS' if navigator.goal_found else 'DISCOVER'} | "
                f"mean_eps={navigator.mean_epsilon:.4f} | "
                f"dangers={len(navigator.danger_cells)} | goal={navigator.goal_cell}",
                f"Graph: known_edges={edges_count} | known_walls={walls_count} | "
                f"visits={navigator._total_visits} | replay_buf={len(navigator._replay_buf)}",
                "Moves: " + " | ".join(f"{a}={action_tally[a]}" for a in ACTIONS),
                "Top V(s):    " + "  ".join(f"{c}:{v:.3f}" for c, v in ranked[:5]),
                "Bottom V(s): " + "  ".join(f"{c}:{v:.3f}" for c, v in ranked[-5:]),
            ]

            print("\n" + "─" * 72)
            for line in summary_lines:
                print(line)
            print("─" * 72 + "\n")

            _write_vgrid(
                log_path, navigator, target, total_steps, episode,
                navigator.goal_found, summary_lines, step_buffer,
                is_continuation=is_cont,
            )

        pos = new_pos
        time.sleep(MOVE_DELAY)

    return goal_eps_this_session


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────

def main():
    while True:
        print("\n1. Enter World")
        print("2. Manual Play")
        print("3. Auto (Q-Learning) — single world")
        print("4. Reset")
        print("5. Exit")

        choice = input("Choose: ").strip()

        if choice == "1":
            _enter_world(input("World ID: ").strip())

        elif choice == "2":
            manual_mode(input("World ID: ").strip())

        elif choice == "3":
            auto_mode(input("World ID: ").strip())

        elif choice == "4":
            print(reset())

        elif choice == "5":
            break

        else:
            print("Invalid — pick 1-6.")


if __name__ == "__main__":
    main()
