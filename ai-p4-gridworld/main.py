import os
import time
from api import enter_world, get_location, move, reset
from agent import GridNavigator, MOVES as ACTIONS, GOAL_REWARD_THRESHOLD, TRAP_REWARD_THRESHOLD


def _parse_pos(s: dict) -> tuple:
    return int(s["x"]), int(s["y"])


def _location_info() -> tuple:
    """Return (world_str, pos_tuple) from the location API, or (None, None)."""
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
    """Enter world; treat 'already in this world' as success."""
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


def _await_valid_pos(target_world, retries: int = 5) -> tuple:
    """Retry until we get a valid position inside target_world."""
    for attempt in range(retries):
        w, pos = _location_info()
        if str(w) == str(target_world) and pos is not None:
            return pos
        print(f"[Retry {attempt+1}/{retries}] world={w}, pos={pos}")
        time.sleep(2)
    return None


def _write_vgrid(log_path, navigator, world_id, total_steps, episode,
                 is_exploit, summary_lines, step_buffer, is_continuation=False):
    """Overwrite the log file with the current V grid and recent steps."""
    q = navigator.q_table
    rows = min(q.shape[0], 40)
    cols = min(q.shape[1], 40)
    run_label = "CONTINUED" if is_continuation else "NEW"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== World {world_id} | Step {total_steps} | Ep {episode} | "
                f"Phase: {'EXPLOIT' if is_exploit else 'DISCOVER'} | {run_label} ===\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\nV-VALUE GRID (max Q per cell)  [col = X axis, row = Y axis]:\n")
        f.write("     " + "".join(f"{c:8}" for c in range(cols)) + "\n")
        for r in range(rows):
            f.write(f"{r:4} " + "".join(f"{float(q[r][c].max()):8.2f}" for c in range(cols)) + "\n")
        f.write("\n--- Recent Steps ---\n")
        for s in step_buffer:
            f.write(s + "\n")


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


def auto_mode(world_id):
    target = str(world_id)
    if not _enter_world(target):
        return

    navigator = GridNavigator(target)
    navigator.load()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"world_{target}.txt")
    step_buffer: list = []
    STEP_BUFFER = 60

    pos = _await_valid_pos(target)
    if pos is None:
        print("[Error] Could not get a valid starting position.")
        return

    # ── session continuity ────────────────────────────────────────────
    is_cont = navigator.is_continuation
    sess    = navigator.session_stats

    # ── counters ──────────────────────────────────────────────────────
    step      = 0
    LOG_EVERY = 20
    recent_rewards: list = []
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

    def _sync_session():
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

    print(f"\n{'='*60}")
    print(f"  World {target}  |  Phase: "
          f"{'EXPLOIT (goal known)' if navigator.goal_found else 'DISCOVER'}")
    if is_cont and sess:
        print(f"  [CONTINUED] from step {total_steps} | episode {episode}")
    else:
        print(f"  [NEW RUN]")
    print(f"{'='*60}\n")

    while True:
        step        += 1
        total_steps += 1

        action_idx, phase = navigator.pick_action(pos)
        direction = ACTIONS[action_idx]
        action_tally[direction] += 1
        if phase == "discover":
            discover_steps += 1
        else:
            exploit_steps += 1

        x, y    = pos
        old_q   = float(navigator.q_table[x][y][action_idx])
        res     = move(target, direction)

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

        # ── terminal state detection ──────────────────────────────────
        if new_state_raw is None:
            navigator.record_terminal(pos, action_idx, reward)

            if reward >= GOAL_REWARD_THRESHOLD:
                print(f"\n:trophy: [Episode {episode}] GOAL REACHED at {pos} "
                      f"| reward={reward:.1f} | run={run_id}")
                navigator.flag_goal(pos)

            elif reward <= TRAP_REWARD_THRESHOLD:
                print(f"\n:skull: [Episode {episode}] TRAP at {pos} "
                      f"| reward={reward:.1f} | run={run_id}")
                navigator.flag_danger(pos)

            else:
                print(f"\n:warning:  [Episode {episode}] Episode ended at {pos} "
                      f"| reward={reward:.1f} | run={run_id}")

            _sync_session()
            navigator.save()

            # Re-enter and start fresh episode
            if not _enter_world(target):
                print("Failed to re-enter world. Stopping.")
                break

            new_pos = _await_valid_pos(target)
            if new_pos is None:
                print("[Error] Could not get valid position after re-entry. Stopping.")
                break

            episode += 1
            step     = 1

            print(f"\n--- Episode {episode} start | "
                  f"Phase: {'EXPLOIT' if navigator.goal_found else 'DISCOVER'} | "
                  f"pos={new_pos} ---\n")

            pos = new_pos
            continue

        # ── normal step ───────────────────────────────────────────────
        new_pos = _parse_pos(new_state_raw)
        if new_pos == pos:
            stall_steps += 1
        seen_cells.add(new_pos)

        nx, ny      = new_pos
        best_next   = float(navigator.q_table[nx][ny].max())
        td_target   = reward + navigator.gamma * best_next
        td_error    = td_target - old_q

        navigator.record_transition(pos, action_idx, reward, new_pos)
        new_q = float(navigator.q_table[x][y][action_idx])
        _sync_session()
        navigator.save()

        step_line = (
            f"Run {run_id} | Ep {episode} | Step {step} | "
            f"{pos} -> {new_pos} | {direction} ({phase}) | r={reward:.3f} | "
            f"dQ={td_error:+.4f} | newQ={new_q:.4f}"
        )
        print(step_line)
        step_buffer.append(step_line)
        if len(step_buffer) > STEP_BUFFER:
            step_buffer.pop(0)

        # ── periodic summary ──────────────────────────────────────────
        if total_steps % LOG_EVERY == 0:
            avg_all    = total_reward / total_steps
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            stall_rate = stall_steps / total_steps

            ranked = sorted(
                ((c, float(navigator.q_table[c[0]][c[1]].max())) for c in seen_cells),
                key=lambda t: t[1], reverse=True
            )

            summary_lines = [
                f"Summary @{total_steps} steps | "
                f"avgR(all)={avg_all:.4f} | avgR(recent)={avg_recent:.4f} | "
                f"stall={stall_rate:.1%} | cells={len(seen_cells)}",
                f"Phase: {'EXPLOIT' if navigator.goal_found else 'DISCOVER'} | "
                f"mean_eps={navigator.mean_epsilon:.4f} | "
                f"dangers={len(navigator.danger_cells)} | goal={navigator.goal_cell} | "
                f"total_visits={navigator._total_visits}",
                "Moves: " + " | ".join(f"{a}={action_tally[a]}" for a in ACTIONS),
                "Top V(s):    " + "  ".join(f"{c}:{v:.3f}" for c, v in ranked[:5]),
                "Bottom V(s): " + "  ".join(f"{c}:{v:.3f}" for c, v in ranked[-5:]),
            ]

            print("\n" + "─" * 72)
            for line in summary_lines:
                print(line)
            print("─" * 72 + "\n")

            _write_vgrid(log_path, navigator, target, total_steps, episode,
                         navigator.goal_found, summary_lines, step_buffer,
                         is_continuation=is_cont)

        pos = new_pos
        time.sleep(2)


def main():
    while True:
        print("\n1. Enter World")
        print("2. Manual Play")
        print("3. Auto (Q-Learning)")
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
            print("Invalid — pick 1-5.")


if __name__ == "__main__":
    main()