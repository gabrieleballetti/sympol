import numpy as np
import random
import collections
import pickle

# --- Import your actual Polytope Class ---
from sympol.polytope import Polytope

PolytopeClass = Polytope


# --- Helper Functions for Log-Concavity ---
def trim_trailing_zeros(seq):
    """Removes trailing zeros from a list/tuple of numbers."""
    if not seq:
        return []
    idx = len(seq) - 1
    while idx >= 0 and seq[idx] == 0:
        idx -= 1
    return list(seq[: idx + 1])  # Return a list


def calculate_min_log_concavity_difference(h_star_vector):
    """
    Calculates the minimum of (h_i^2 - h_{i-1}*h_{i+1}) for the h_star_vector.
    Returns the minimum difference, or None if not applicable (e.g., vector too short).
    A negative value indicates a failure of log-concavity.
    """
    h_trimmed = trim_trailing_zeros(h_star_vector)

    if len(h_trimmed) < 3:
        return None  # Log-concavity check not applicable or trivially satisfied

    min_diff = float("inf")

    for i in range(1, len(h_trimmed) - 1):
        h_i = h_trimmed[i]
        h_prev = h_trimmed[i - 1]
        h_next = h_trimmed[i + 1]

        # Ensure all are non-negative as expected for h* vectors
        if not (
            isinstance(h_i, (int, float))
            and h_i >= 0
            and isinstance(h_prev, (int, float))
            and h_prev >= 0
            and isinstance(h_next, (int, float))
            and h_next >= 0
        ):
            # This case should ideally not happen if h_star_vector is clean
            # print(f"Warning: Unexpected elements in h_trimmed for log-concavity check: {[h_prev, h_i, h_next]}")
            continue  # Skip this triplet if data is not as expected

        diff = h_i**2 - (h_prev * h_next)
        min_diff = min(min_diff, diff)

    return min_diff if min_diff != float("inf") else None


class PolytopeEnv:
    def __init__(
        self, dimension, num_extra_vectors, max_steps_per_episode=200, K_lc_progress=5.0
    ):
        self.dimension = dimension
        self.num_extra_vectors = num_extra_vectors
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.K_lc_progress = (
            K_lc_progress  # Hyperparameter for rewarding closeness to non-log-concavity
        )

        self.origin = [0] * self.dimension
        self.basis_vectors = []
        for i in range(self.dimension):
            vec = [0] * self.dimension
            vec[i] = 1
            self.basis_vectors.append(vec)

        self.action_space_size = self.num_extra_vectors * self.dimension
        self.current_extra_vectors = self._generate_initial_extra_vectors()

    def _generate_initial_extra_vectors(self):
        return [
            [random.randint(0, 1) for _ in range(self.dimension)]
            for _ in range(self.num_extra_vectors)
        ]

    def _get_full_points(self, extra_vectors):
        return [self.origin] + self.basis_vectors + extra_vectors

    def _get_state_representation(self, extra_vectors):
        return tuple(map(tuple, extra_vectors))

    def reset(self):
        self.current_step = 0
        self.current_extra_vectors = self._generate_initial_extra_vectors()
        return self._get_state_representation(self.current_extra_vectors)

    def step(self, action_index):
        self.current_step += 1
        extra_vector_idx = action_index // self.dimension
        bit_to_flip_idx = action_index % self.dimension

        next_extra_vectors = [list(vec) for vec in self.current_extra_vectors]
        next_extra_vectors[extra_vector_idx][bit_to_flip_idx] = (
            1 - next_extra_vectors[extra_vector_idx][bit_to_flip_idx]
        )
        self.current_extra_vectors = next_extra_vectors

        current_full_points = self._get_full_points(self.current_extra_vectors)
        next_state_representation = self._get_state_representation(
            self.current_extra_vectors
        )

        current_info = {"points": current_full_points}
        reward = 0
        done = False

        try:
            polytope_obj = PolytopeClass(current_full_points)
        except Exception as e:
            error_msg = str(e)
            # print(f"    ERROR creating Polytope object: {e} for points: {current_full_points}") # Can be very verbose
            reward = -100.0
            done = True
            current_info["error"] = error_msg
            return next_state_representation, reward, done, current_info

        is_idp = polytope_obj.is_idp
        h_star_for_info = None
        min_lc_diff_for_info = None

        if not is_idp:
            reward = -10.0
        else:  # Is IDP
            try:
                # Ensure h_star_vector is a list of numbers as expected by helper functions
                h_star = list(polytope_obj.h_star_vector)
                current_info["h_star"] = h_star  # Store for info regardless of outcome
                h_star_for_info = h_star  # For more specific logging if needed
            except AttributeError:
                # print(f"Warning: Polytope object for {current_full_points} does not have h_star_vector attribute.")
                reward = (
                    -5.0
                )  # Penalize if h_star_vector is missing for an IDP polytope
                h_star = None  # Cannot proceed with log-concavity check
            except (
                TypeError
            ):  # If h_star_vector is not iterable or elements not numbers
                # print(f"Warning: h_star_vector for {current_full_points} is not a list of numbers.")
                reward = -5.0
                h_star = None

            if h_star is not None:  # Proceed only if h_star was successfully obtained
                min_lc_diff = calculate_min_log_concavity_difference(h_star)
                current_info["min_lc_diff"] = min_lc_diff  # Store for info
                min_lc_diff_for_info = min_lc_diff

                if min_lc_diff is None:  # Too short to check, or all zeros after trim
                    reward = 1.0  # Base IDP reward (trivially log-concave)
                elif min_lc_diff < 0:  # Counterexample found!
                    reward = 100.0
                    done = True
                    print(
                        f"    SUCCESS! Counterexample found: dim={self.dimension}, n_extra={self.num_extra_vectors}, min_lc_diff={min_lc_diff:.2f}"
                    )
                    print(f"    Extra Vectors: {self.current_extra_vectors}")
                    print(f"    h* vector: {h_star_for_info}")
                else:  # min_lc_diff >= 0 (IDP and log-concave)
                    base_idp_reward = 1.0
                    proximity_reward = self.K_lc_progress / (1.0 + min_lc_diff)
                    reward = base_idp_reward + proximity_reward
            # else: if h_star was None, reward is already -5.0 or as set by IDP check

        # Apply step penalty unless it was a major Polytope creation error
        if reward > -90:  # Don't subtract from the -100 error
            reward -= 0.1

        if self.current_step >= self.max_steps_per_episode and not done:
            done = True

        return next_state_representation, reward, done, current_info


class QLearningAgent:
    def __init__(
        self,
        action_space_size,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate_initial=1.0,
        exploration_decay=0.9995,
        exploration_min=0.05,
    ):  # Adjusted decay/min
        self.q_table = collections.defaultdict(lambda: np.zeros(action_space_size))
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate_initial
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min

    def get_action(self, state_representation):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state_representation])

    def learn(self, state_repr, action_idx, reward, next_state_repr, done):
        current_q = self.q_table[state_repr][action_idx]
        max_future_q = np.max(self.q_table[next_state_repr]) if not done else 0.0
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_repr][action_idx] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath="q_table.pkl"):
        try:
            with open(filepath, "wb") as f:
                pickle.dump(dict(self.q_table), f)
            print(f"Q-table saved to {filepath}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, filepath="q_table.pkl"):
        try:
            with open(filepath, "rb") as f:
                loaded_q_dict = pickle.load(f)
            self.q_table = collections.defaultdict(
                lambda: np.zeros(self.action_space_size)
            )
            self.q_table.update(loaded_q_dict)
            print(f"Q-table loaded from {filepath}")
        except FileNotFoundError:
            print(f"No Q-table found at {filepath}, starting fresh.")
        except Exception as e:
            print(f"Error loading Q-table: {e}")


# --- Main Training Loop ---
def train(
    dimensions_to_try,
    num_extra_vectors_to_try,
    num_episodes,
    max_steps_per_episode=200,
    K_lc_progress_env=5.0,
):
    found_counterexamples = []

    for d in dimensions_to_try:
        print(f"\n--- Searching in Dimension {d} ---")
        for n_extra in num_extra_vectors_to_try:
            print(f"  -- Using {n_extra} Extra Vector(s) --")

            env = PolytopeEnv(
                dimension=d,
                num_extra_vectors=n_extra,
                max_steps_per_episode=max_steps_per_episode,
                K_lc_progress=K_lc_progress_env,
            )
            agent = QLearningAgent(action_space_size=env.action_space_size)

            q_table_filename = f"q_table_d{d}_ne{n_extra}_klc{K_lc_progress_env}.pkl"
            agent.load_q_table(q_table_filename)
            episode_rewards = []

            for episode in range(num_episodes):
                state_repr = env.reset()
                total_reward_episode = 0.0

                for step in range(max_steps_per_episode):
                    action = agent.get_action(state_repr)
                    next_state_repr, reward, done, info = env.step(action)
                    agent.learn(state_repr, action, reward, next_state_repr, done)
                    state_repr = next_state_repr
                    total_reward_episode += reward

                    if done:
                        if "error" in info:
                            # print(f"Ep {episode+1}: Ended due to Polytope error. Reward: {total_reward_episode:.2f}")
                            pass
                        elif (
                            reward > 90
                        ):  # Successful counterexample found (reward approx 100 - 0.1)
                            found_counterexamples.append(
                                {
                                    "dimension": d,
                                    "num_extra_vectors": n_extra,
                                    "points": info.get("points"),
                                    "extra_vectors_only": env.current_extra_vectors,  # Access from env instance
                                    "episode": episode + 1,
                                    "steps": step + 1,
                                    "h_star_vector": info.get("h_star"),
                                    "min_lc_diff": info.get("min_lc_diff"),
                                }
                            )
                            # Success message already printed by env.step
                        break

                episode_rewards.append(total_reward_episode)
                agent.decay_epsilon()

                if (
                    episode + 1
                ) % 200 == 0:  # Log progress less frequently for longer runs
                    avg_reward = (
                        np.mean(episode_rewards[-200:]) if episode_rewards else 0
                    )
                    print(
                        f"  Dim {d}, N_extra {n_extra} - Ep {episode+1}/{num_episodes}. Avg Reward (last 200): {avg_reward:.2f}. Eps: {agent.epsilon:.3f}"
                    )

            agent.save_q_table(q_table_filename)

    if found_counterexamples:
        print("\n--- Found Counterexamples ---")
        for i, ex in enumerate(found_counterexamples):
            print(
                f"  Ex {i+1}: Dim={ex['dimension']}, N_extra={ex['num_extra_vectors']}, Ep={ex['episode']}, Steps={ex['steps']}"
            )
            print(f"    Extra Vecs: {ex['extra_vectors_only']}")
            print(
                f"    h* vector: {ex['h_star_vector']}, Min LC Diff: {ex['min_lc_diff']:.2f}"
            )

        counterexamples_filename = f"found_counterexamples_klc{K_lc_progress_env}.pkl"
        try:
            with open(counterexamples_filename, "wb") as f:
                pickle.dump(found_counterexamples, f)
            print(f"Counterexamples saved to {counterexamples_filename}")
        except Exception as e:
            print(f"Error saving counterexamples: {e}")
    else:
        print("\n--- No counterexamples found with current settings. ---")
    return found_counterexamples


if __name__ == "__main__":
    dims = list(range(7, 9))  # Example: 7 to 8. For 7-15: list(range(7,16))
    num_extras = list(range(1, 3))  # Example: 1 to 2. For 1-3: list(range(1,4))

    episodes = 2000  # Increase for thorough search
    steps_per_episode = 300  # Allow more steps
    K_lc_param = 5.0  # Hyperparameter for rewarding "closeness" to non-log-concavity

    print("Starting RL training with refined rewards...")
    print(f"Using Polytope class: {PolytopeClass.__name__} from sympol.polytope")
    print(f"K_lc_progress parameter for reward shaping: {K_lc_param}")

    try:
        test_points = (
            [[0] * dims[0]]
            + [[1 if i == j else 0 for i in range(dims[0])] for j in range(dims[0])]
            + [
                [random.randint(0, 1) for _ in range(dims[0])]
                for _ in range(num_extras[0])
            ]
        )
        p_test = PolytopeClass(test_points)
        _ = p_test.is_idp
        _ = p_test.h_star_vector  # Check if attribute exists
        print("Successfully created a test Polytope object and accessed attributes.")
    except Exception as e:
        print(f"ERROR: Could not initialize/use your Polytope class as expected: {e}")
        print(
            "Ensure 'sympol.polytope' is installed, Polytope(points) works, and has .is_idp and .h_star_vector attributes."
        )
        exit()

    found_examples = train(
        dimensions_to_try=dims,
        num_extra_vectors_to_try=num_extras,
        num_episodes=episodes,
        max_steps_per_episode=steps_per_episode,
        K_lc_progress_env=K_lc_param,
    )

    # ... (rest of the summary printing remains similar) ...
    if found_examples:
        print("\n--- Summary of Found Counterexamples ---")
        for i, ex in enumerate(found_examples):
            print(f"\n  Counterexample {i+1}:")
            print(
                f"    Dimension: {ex['dimension']}, Num Extra Vectors: {ex['num_extra_vectors']}"
            )
            print(f"    Found in Episode: {ex['episode']}, Step: {ex['steps']}")
            print(f"    Extra Vectors Config: {ex['extra_vectors_only']}")
            print(f"    h* vector: {ex['h_star_vector']}")
            print(
                f"    Min Log-Concavity Difference: {ex.get('min_lc_diff', 'N/A'):.2f}"
                if isinstance(ex.get("min_lc_diff"), (int, float))
                else f"    Min Log-Concavity Difference: {ex.get('min_lc_diff', 'N/A')}"
            )
            # To print all points (can be very long):
            # print(f"    Full list of points defining the polytope:")
            # for p_idx, pt in enumerate(ex['points']): print(f"      {pt}")
    else:
        print("\n--- No counterexamples were found in this run. ---")
