import time
from api import enter_world, get_location, move, reset
from agent import QLearningAgent, ACTIONS


def parse_state(s):
    return int(s["x"]), int(s["y"])


def get_current_state():
    loc = get_location()
    if loc.get("code") != "OK":
        print(loc)
        return None

    try:
        x_str, y_str = loc["state"].split(":")
        return int(x_str), int(y_str)
    except (KeyError, ValueError):
        print({"code": "FAIL", "message": f"Unexpected location payload: {loc}"})
        return None


def get_location_info():
    """Return world and state tuple from location API, or None values if not usable."""
    loc = get_location()
    if loc.get("code") != "OK":
        print(loc)
        return None, None

    world = str(loc.get("world", ""))
    state_raw = loc.get("state", "")
    if not state_raw:
        return world, None

    try:
        x_str, y_str = state_raw.split(":")
        return world, (int(x_str), int(y_str))
    except (ValueError, TypeError):
        print({"code": "FAIL", "message": f"Unexpected location payload: {loc}"})
        return world, None


def ensure_world(world_id):
    """Try to enter world; treat 'already in this world' as success."""
    world_id = str(world_id)
    res = enter_world(world_id)
    if res.get("code") == "OK":
        print(res)
        return True

    message = str(res.get("message", ""))
    if "currently in world" in message and world_id in message:
        print(f"Already in world {world_id}. Continuing.")
        return True

    print(res)
    return False


def manual_mode(world_id):
    if not ensure_world(world_id):
        return

    while True:
        move_dir = input("Move (N/E/S/W or q): ").upper()
        if move_dir == "Q":
            break

        if move_dir not in ACTIONS:
            print("Invalid move. Use N, E, S, W, or Q.")
            continue

        res = move(world_id, move_dir)
        print(res)


def auto_mode(world_id):
    target_world = str(world_id)
    if not ensure_world(world_id):
        return

    agent = QLearningAgent(target_world)
    state = get_current_state()
    if state is None:
        return

    episode = 1
    step = 0
    total_steps = 0
    total_reward = 0.0
    recent_rewards = []
    action_counts = {a: 0 for a in ACTIONS}
    explore_steps = 0
    exploit_steps = 0
    same_state_steps = 0
    visited_states = set([state])
    summary_every = 20

    while True:
        step += 1
        total_steps += 1

        action_idx, mode = agent.choose_action_with_mode(state)
        action = ACTIONS[action_idx]
        action_counts[action] += 1
        if mode == "explore":
            explore_steps += 1
        else:
            exploit_steps += 1

        x, y = state
        old_q = float(agent.q_table[x][y][action_idx])

        res = move(target_world, action)

        if res.get("code") != "OK":
            print(res)
            break

        reward = float(res.get("reward", 0.0))
        total_reward += reward
        recent_rewards.append(reward)
        if len(recent_rewards) > summary_every:
            recent_rewards.pop(0)

        run_id = res.get("runId", "?")
        move_world = str(res.get("worldId", target_world))
        new_state_payload = res.get("newState")

        if new_state_payload is not None:
            new_state = parse_state(new_state_payload)
        else:
            # If the move ended or dropped us outside a world, recover from location API.
            loc_world, loc_state = get_location_info()
            if loc_world != target_world or loc_state is None:
                print(
                    f"Episode {episode} ended/outside world (location world={loc_world}, state={loc_state}). Re-entering world {target_world}."
                )
                if not ensure_world(target_world):
                    print("Failed to re-enter target world. Stopping auto mode.")
                    break
                loc_world, loc_state = get_location_info()
                if loc_world != target_world or loc_state is None:
                    print(
                        {
                            "code": "FAIL",
                            "message": "Could not obtain valid state after re-entering world.",
                            "locationWorld": loc_world,
                            "locationState": loc_state,
                        }
                    )
                    break
                episode += 1
                step = 1

            new_state = loc_state

        if new_state == state:
            same_state_steps += 1
        visited_states.add(new_state)

        nx, ny = new_state
        best_next = float(agent.q_table[nx][ny].max())
        td_target = reward + agent.gamma * best_next
        td_error = td_target - old_q
        state_value_before = float(agent.q_table[x][y].max())
        agent.update(state, action_idx, reward, new_state)
        new_q = float(agent.q_table[x][y][action_idx])
        state_value_after = float(agent.q_table[x][y].max())
        agent.save_q()

        print(
            f"Run {run_id} | Episode {episode} | Step {step} | MoveWorld {move_world} | "
            f"State {state} -> {new_state} | Action {action} ({mode}) | Reward {reward:.3f}"
        )
        print(
            f"Q-update: oldQ={old_q:.4f}, bestNext={best_next:.4f}, tdTarget={td_target:.4f}, "
            f"tdError={td_error:.4f}, newQ={new_q:.4f}, V(s)_before={state_value_before:.4f}, V(s)_after={state_value_after:.4f}"
        )

        if total_steps % summary_every == 0:
            avg_reward_all = total_reward / total_steps
            avg_reward_recent = sum(recent_rewards) / len(recent_rewards)
            explore_rate = explore_steps / total_steps
            stall_rate = same_state_steps / total_steps

            ranked_states = []
            for sx, sy in visited_states:
                ranked_states.append(((sx, sy), float(agent.q_table[sx][sy].max())))
            ranked_states.sort(key=lambda item: item[1], reverse=True)

            top_states = ranked_states[:5]
            bottom_states = ranked_states[-5:]

            print("-" * 88)
            print(
                f"Learning Summary @ step {total_steps}: avgRewardAll={avg_reward_all:.4f}, "
                f"avgRewardRecent={avg_reward_recent:.4f}, exploreRate={explore_rate:.2%}, "
                f"stallRate={stall_rate:.2%}, uniqueStates={len(visited_states)}"
            )
            print(
                "Action usage: "
                + ", ".join([f"{a}={action_counts[a]}" for a in ACTIONS])
            )
            print(
                "Top learned states V(s): "
                + ", ".join([f"{st}:{val:.3f}" for st, val in top_states])
            )
            print(
                "Most avoided states V(s): "
                + ", ".join([f"{st}:{val:.3f}" for st, val in bottom_states])
            )
            print("-" * 88)

        state = new_state

        time.sleep(0.2)  # API limit


def main():
    while True:
        print("\n1. Enter World")
        print("2. Manual Play")
        print("3. Auto (Q-Learning)")
        print("4. Reset")
        print("5. Exit")

        choice = input("Choose: ")

        if choice == "1":
            w = input("World ID: ")
            ensure_world(w)

        elif choice == "2":
            w = input("World ID: ")
            manual_mode(w)

        elif choice == "3":
            w = input("World ID: ")
            auto_mode(w)

        elif choice == "4":
            print(reset())

        elif choice == "5":
            break

        else:
            print("Invalid choice. Pick 1-5.")


if __name__ == "__main__":
    main()