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

    pos = _await_valid_pos(target)
    if pos is None:
        print("[Error] Could not get a valid starting position.")
        return

    # ── counters ──────────────────────────────────────────────────────
    episode      = 1
    step         = 0
    total_steps  = 0
    total_reward = 0.0
    recent_rewards: list = []
    action_tally = {a: 0 for a in ACTIONS}
    discover_steps = 0
    exploit_steps  = 0
    stall_steps    = 0
    seen_cells: set = {pos}
    LOG_EVERY = 20

    print(f"\n{'='*60}")
    print(f"  World {target}  |  Phase: "
          f"{'EXPLOIT (goal known)' if navigator.goal_found else 'DISCOVER'}")
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
        navigator.save()

        print(
            f"Run {run_id} | Ep {episode} | Step {step} | "
            f"{pos} → {new_pos} | {direction} ({phase}) | r={reward:.3f} | "
            f"ΔQ={td_error:+.4f} | newQ={new_q:.4f}"
        )

        # ── periodic summary ──────────────────────────────────────────
        if total_steps % LOG_EVERY == 0:
            avg_all    = total_reward / total_steps
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            stall_rate = stall_steps / total_steps

            ranked = sorted(
                ((c, float(navigator.q_table[c[0]][c[1]].max())) for c in seen_cells),
                key=lambda t: t[1], reverse=True
            )

            print("\n" + "─" * 72)
            print(
                f"Summary @{total_steps} steps | "
                f"avgR(all)={avg_all:.4f} | avgR(recent)={avg_recent:.4f} | "
                f"stall={stall_rate:.1%} | cells={len(seen_cells)}"
            )
            print(
                f"Phase: {'EXPLOIT' if navigator.goal_found else 'DISCOVER'} | "
                f"mean_eps={navigator.mean_epsilon:.4f} | "
                f"dangers={len(navigator.danger_cells)} | goal={navigator.goal_cell}"
            )
            print("Moves: " + " | ".join(f"{a}={action_tally[a]}" for a in ACTIONS))
            print("Top V(s):    " + "  ".join(f"{c}:{v:.3f}" for c, v in ranked[:5]))
            print("Bottom V(s): " + "  ".join(f"{c}:{v:.3f}" for c, v in ranked[-5:]))
            print("─" * 72 + "\n")

        pos = new_pos
        time.sleep(0.2)


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