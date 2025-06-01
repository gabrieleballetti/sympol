import numpy as np
import random
import collections
import pickle  # For saving/loading Q-table and found examples

# --- Import your actual Polytope Class ---
from sympol.polytope import Polytope

# The PolytopeClass to be used by the environment
PolytopeClass = Polytope


class PolytopeEnv:
    def __init__(self, dimension, num_extra_vectors, max_steps_per_episode=200):
        self.dimension = dimension
        self.num_extra_vectors = num_extra_vectors
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        # Fixed points: origin and standard basis vectors
        self.origin = [0] * self.dimension
        self.basis_vectors = []
        for i in range(self.dimension):
            vec = [0] * self.dimension
            vec[i] = 1
            self.basis_vectors.append(vec)

        self.action_space_size = self.num_extra_vectors * self.dimension
        # The state will be the list of 'extra' vectors
        self.current_extra_vectors = self._generate_initial_extra_vectors()

    def _generate_initial_extra_vectors(self):
        extra_vectors = []
        for _ in range(self.num_extra_vectors):
            extra_vectors.append([random.randint(0, 1) for _ in range(self.dimension)])
        return extra_vectors

    def _get_full_points(self, extra_vectors):
        return [self.origin] + self.basis_vectors + extra_vectors

    def _get_state_representation(self, extra_vectors):
        # Convert list of lists to tuple of tuples for hashability (used as dict keys)
        return tuple(map(tuple, extra_vectors))

    def reset(self):
        self.current_step = 0
        self.current_extra_vectors = self._generate_initial_extra_vectors()
        return self._get_state_representation(self.current_extra_vectors)

    def step(self, action_index):
        self.current_step += 1

        # Decode action: (extra_vector_idx, bit_to_flip_idx)
        extra_vector_idx = action_index // self.dimension
        bit_to_flip_idx = action_index % self.dimension

        # Apply action (create a new state by modifying a copy of current_extra_vectors)
        next_extra_vectors = [
            list(vec) for vec in self.current_extra_vectors
        ]  # Deep copy
        next_extra_vectors[extra_vector_idx][bit_to_flip_idx] = (
            1 - next_extra_vectors[extra_vector_idx][bit_to_flip_idx]
        )

        self.current_extra_vectors = (
            next_extra_vectors  # Update current state's extra vectors
        )

        current_full_points = self._get_full_points(self.current_extra_vectors)

        # --- Use your actual PolytopeClass ---
        try:
            polytope_obj = PolytopeClass(current_full_points)
        except Exception as e:
            print(f"    ERROR creating Polytope object: {e}")
            print(f"    Points causing error: {current_full_points}")
            # Penalize heavily and end episode if Polytope creation fails
            reward = -100.0
            done = True
            next_state_representation = self._get_state_representation(
                self.current_extra_vectors
            )
            return (
                next_state_representation,
                reward,
                done,
                {"points": current_full_points, "error": str(e)},
            )

        reward = 0
        done = False

        # Check properties
        is_idp = polytope_obj.is_idp
        has_log_concave = polytope_obj.has_log_concave_h_star_vector

        if not is_idp:
            reward = -10.0  # Penalize for not being IDP
        else:  # Is IDP
            if not has_log_concave:
                reward = 100.0  # Goal! Counterexample found
                done = True
                print(
                    f"    SUCCESS! Counterexample found: dim={self.dimension}, n_extra={self.num_extra_vectors}"
                )
                print(f"    Extra Vectors: {self.current_extra_vectors}")
            else:  # IDP, but log-concave (or property not met for non-log-concave)
                reward = 1.0  # Small reward for being IDP

        reward -= 0.1  # Step penalty to encourage efficiency

        if self.current_step >= self.max_steps_per_episode:
            done = True
            if reward < 90:  # Not already a success from finding the counterexample
                # print(f"    Episode ended due to max steps. Dim={self.dimension}, N_extra={self.num_extra_vectors}") # Can be verbose
                pass

        next_state_representation = self._get_state_representation(
            self.current_extra_vectors
        )
        return next_state_representation, reward, done, {"points": current_full_points}


class QLearningAgent:
    def __init__(
        self,
        action_space_size,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate_initial=1.0,
        exploration_decay=0.999,
        exploration_min=0.01,
    ):
        # Q-table: state_representation -> array of Q-values for actions
        self.q_table = collections.defaultdict(lambda: np.zeros(action_space_size))
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate_initial
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min

    def get_action(self, state_representation):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value
            # defaultdict ensures state_representation is in q_table with zeros if new
            return np.argmax(self.q_table[state_representation])

    def learn(self, state_repr, action_idx, reward, next_state_repr, done):
        current_q = self.q_table[state_repr][action_idx]

        # defaultdict ensures next_state_repr is in q_table for max Q calculation
        max_future_q = np.max(self.q_table[next_state_repr]) if not done else 0.0

        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_repr][action_idx] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath="q_table.pkl"):
        try:
            with open(filepath, "wb") as f:
                pickle.dump(
                    dict(self.q_table), f
                )  # Convert defaultdict to dict for pickling
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
    dimensions_to_try, num_extra_vectors_to_try, num_episodes, max_steps_per_episode=200
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
            )
            agent = QLearningAgent(action_space_size=env.action_space_size)

            # Optional: Load Q-table if continuing a previous run
            q_table_filename = f"q_table_d{d}_ne{n_extra}.pkl"
            agent.load_q_table(q_table_filename)

            episode_rewards = []  # For tracking progress

            for episode in range(num_episodes):
                print(
                    f"\n  Episode {episode + 1}/{num_episodes} - Dim {d}, N_extra {n_extra}"
                )
                state_repr = env.reset()
                total_reward_episode = 0.0

                for step in range(max_steps_per_episode):
                    print(".", end="", flush=True)
                    action = agent.get_action(state_repr)
                    next_state_repr, reward, done, info = env.step(action)

                    agent.learn(state_repr, action, reward, next_state_repr, done)

                    state_repr = next_state_repr
                    total_reward_episode += reward

                    if done:
                        if "error" in info:  # Polytope creation failed
                            print(
                                f"Episode {episode+1}/{num_episodes}: Ended due to Polytope error. Total Reward: {total_reward_episode:.2f}"
                            )
                        elif (
                            reward > 90
                        ):  # Successful counterexample found (reward == 100 - 0.1 step penalty)
                            current_points = info["points"]
                            found_counterexamples.append(
                                {
                                    "dimension": d,
                                    "num_extra_vectors": n_extra,
                                    "points": current_points,  # Save the full list of points
                                    "extra_vectors_only": env.current_extra_vectors,  # Save just the modifiable part
                                    "episode": episode + 1,
                                    "steps": step + 1,
                                }
                            )
                            print(
                                f"Episode {episode+1}/{num_episodes}: Counterexample FOUND! Reward: {total_reward_episode:.2f}, Epsilon: {agent.epsilon:.3f}"
                            )
                            # Optionally, break from episodes for this (d, n_extra) if one is enough
                            # Or continue searching for more diverse examples
                        else:  # Episode ended for other reasons (max steps)
                            # print(f"Episode {episode+1}/{num_episodes}: Ended. Total Reward: {total_reward_episode:.2f}, Epsilon: {agent.epsilon:.3f}")
                            pass
                        break

                episode_rewards.append(total_reward_episode)
                agent.decay_epsilon()

                if (episode + 1) % 100 == 0:  # Log progress
                    avg_reward = (
                        np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    )
                    print(
                        f"  Dim {d}, N_extra {n_extra} - Episode {episode+1}/{num_episodes} complete. Avg Reward (last 100): {avg_reward:.2f}. Epsilon: {agent.epsilon:.3f}"
                    )

            # Save Q-table after all episodes for this (d, n_extra) configuration
            agent.save_q_table(q_table_filename)

    if found_counterexamples:
        print("\n--- Found Counterexamples ---")
        for i, ex in enumerate(found_counterexamples):
            print(
                f"  Example {i+1}: Dim={ex['dimension']}, N_extra={ex['num_extra_vectors']}, Episode={ex['episode']}, Steps={ex['steps']}"
            )
            print(f"    Extra Vectors: {ex['extra_vectors_only']}")
            # Optionally print all points: print(f"    All Points: {ex['points']}")

        counterexamples_filename = "found_counterexamples.pkl"
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
    # Configuration
    # Search in dimension 7 to 15
    # dims = list(range(7, 16))
    dims = [9]  # Start with a smaller range for testing

    # Add 1 to 3 extra vectors
    # num_extras = list(range(1, 4))
    num_extras = [3]  # Start with a smaller range for testing

    # Adjust episodes and steps based on expected computation time of your Polytope class
    episodes = (
        1000  # Number of episodes per (dimension, num_extra_vectors) configuration
    )
    steps_per_episode = 50  # Max steps within one episode

    print("Starting RL training to find counterexamples...")
    print(f"Using Polytope class: {PolytopeClass.__name__} from sympol.polytope")

    # Ensure your sympol library is installed and accessible
    try:
        # A quick test to see if the import worked and class is usable
        test_points = [[0, 0], [1, 0], [0, 1], [1, 1]]
        _ = PolytopeClass(test_points)
        print("Successfully created a test Polytope object.")
    except Exception as e:
        print(f"ERROR: Could not initialize your Polytope class: {e}")
        print(
            "Please ensure 'sympol.polytope' is installed and Polytope(points) works."
        )
        exit()

    found_examples = train(
        dimensions_to_try=dims,
        num_extra_vectors_to_try=num_extras,
        num_episodes=episodes,
        max_steps_per_episode=steps_per_episode,
    )

    if found_examples:
        print("\n--- Summary of Found Counterexamples ---")
        for i, ex in enumerate(found_examples):
            print(f"\n  Counterexample {i+1}:")
            print(f"    Dimension: {ex['dimension']}")
            print(f"    Number of Extra Vectors: {ex['num_extra_vectors']}")
            print(f"    Found in Episode: {ex['episode']}, Step: {ex['steps']}")
            print(f"    Extra Vectors Configuration: {ex['extra_vectors_only']}")
            # To print all points (can be very long):
            # print(f"    Full list of points defining the polytope:")
            # for p_idx, pt in enumerate(ex['points']):
            #      print(f"      {pt}")
    else:
        print("\n--- No counterexamples were found in this run. ---")
