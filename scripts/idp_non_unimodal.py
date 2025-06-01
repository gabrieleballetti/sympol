import numpy as np
import random
import collections
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from sympol.polytope import Polytope


PolytopeClass = Polytope
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")


# --- Helper Function for NORMALIZED Log-Concavity Metric (same as before) ---
def calculate_normalized_min_log_concavity_metric(h_star_vector):
    # ... (no changes from previous version) ...
    if not h_star_vector:
        return None
    try:
        processed_h = [float(x) for x in h_star_vector]
    except (ValueError, TypeError):
        return -float("inf")
    first = 0
    while first < len(processed_h) and processed_h[first] == 0:
        first += 1
    if first == len(processed_h):
        return None
    last = len(processed_h) - 1
    while last > first and processed_h[last] == 0:
        last -= 1
    h_eff = processed_h[first : last + 1]
    if len(h_eff) < 3:
        return None
    has_internal_zero = False
    for k in range(1, len(h_eff) - 1):
        if h_eff[k] == 0:
            has_internal_zero = True
            break
    min_normalized_metric = float("inf")
    calculation_performed = False
    for i in range(1, len(h_eff) - 1):
        h_i, h_prev, h_next = h_eff[i], h_eff[i - 1], h_eff[i + 1]
        current_metric_val = 0.0
        if h_i != 0:
            current_metric_val = (h_i**2 - (h_prev * h_next)) / (h_i**2)
        else:
            if h_prev * h_next > 0:
                current_metric_val = -float("inf")
        min_normalized_metric = min(min_normalized_metric, current_metric_val)
        calculation_performed = True
    if not calculation_performed:
        return None
    if has_internal_zero:
        return (
            min(min_normalized_metric, -0.000001)
            if min_normalized_metric >= 0
            else min_normalized_metric
        )
    return min_normalized_metric


# --- Helper Function for Unimodality ---
def _is_unimodal_internal(
    processed_iterable,
):  # Takes already processed (numeric, trimmed) list
    """Return True if the iterable is unimodal, False otherwise."""
    if (
        not processed_iterable
    ):  # Empty list is arguably unimodal, or not well-defined. Let's say true.
        return True
    if len(processed_iterable) <= 1:  # Single element is unimodal
        return True

    i = 1
    # Phase 1: Non-decreasing
    while (
        i < len(processed_iterable)
        and processed_iterable[i - 1] <= processed_iterable[i]
    ):
        i += 1
    # Phase 2: Non-increasing
    while (
        i < len(processed_iterable)
        and processed_iterable[i - 1] >= processed_iterable[i]
    ):
        i += 1
    return i == len(processed_iterable)


def get_unimodality_status(h_star_vector):
    """
    Checks if the h_star_vector is unimodal after trimming leading/trailing zeros.
    Returns: True if unimodal, False if non-unimodal, None if not applicable (e.g., empty after trim).
    """
    if not h_star_vector:
        return None  # Or True, depending on definition for empty
    try:
        processed_h = [float(x) for x in h_star_vector]
    except (ValueError, TypeError):
        return False  # Treat malformed as non-unimodal for safety

    first = 0
    while first < len(processed_h) and processed_h[first] == 0:
        first += 1
    if first == len(processed_h):
        return None  # All zeros, effectively empty for unimodality check

    last = len(processed_h) - 1
    while last > first and processed_h[last] == 0:
        last -= 1
    h_eff = processed_h[first : last + 1]

    if not h_eff:
        return None  # Should be caught by all_zeros case

    return _is_unimodal_internal(h_eff)


# --- Replay Memory (same) ---
Transition = collections.namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- QNetwork Definition (same) ---
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_factor=3):
        super(QNetwork, self).__init__()
        h1_size = max(128, input_size * hidden_factor)
        h2_size = max(64, h1_size // 2)
        self.fc1 = nn.Linear(input_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, output_size)
        # print(f"QNetwork initialized: In={input_size}, H1={h1_size}, H2={h2_size}, Out={output_size}")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Polytope Environment (UPDATED REWARDS) ---
class PolytopeEnv:
    def __init__(
        self,
        dimension,
        num_extra_vectors,
        max_steps_per_episode=200,
        K_lc_progress=5.0,
        non_idp_penalty=-25.0,
        reward_non_unimodal=200.0,
        reward_non_lc_unimodal=50.0,
    ):  # New reward params
        self.dimension = dimension
        self.num_extra_vectors = num_extra_vectors
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.K_lc_progress = K_lc_progress
        self.state_size = self.dimension * self.num_extra_vectors
        self.action_space_size = self.num_extra_vectors * self.dimension
        self.origin = [0] * self.dimension
        self.basis_vectors = [
            [1 if i == j else 0 for i in range(self.dimension)]
            for j in range(self.dimension)
        ]
        self.current_extra_vectors = self._generate_initial_extra_vectors()
        self.non_idp_penalty = non_idp_penalty
        self.reward_non_unimodal = reward_non_unimodal
        self.reward_non_lc_unimodal = reward_non_lc_unimodal

    def _generate_initial_extra_vectors(self):  # Same
        return [
            [random.randint(0, 1) for _ in range(self.dimension)]
            for _ in range(self.num_extra_vectors)
        ]

    def _get_full_points(self, extra_vectors):  # Same
        return [self.origin] + self.basis_vectors + extra_vectors

    def _get_nn_state_representation(self, extra_vectors):  # Same
        flat_list = [bit for vec in extra_vectors for bit in vec]
        return torch.tensor(flat_list, dtype=torch.float32, device=device).unsqueeze(0)

    def reset(self):  # Same
        self.current_step = 0
        self.current_extra_vectors = self._generate_initial_extra_vectors()
        return self._get_nn_state_representation(self.current_extra_vectors)

    def step(self, action_index):
        self.current_step += 1
        # ... (action decoding and state update - same) ...
        extra_vector_idx = action_index // self.dimension
        bit_to_flip_idx = action_index % self.dimension
        next_extra_vectors_list = [list(vec) for vec in self.current_extra_vectors]
        next_extra_vectors_list[extra_vector_idx][bit_to_flip_idx] = (
            1 - next_extra_vectors_list[extra_vector_idx][bit_to_flip_idx]
        )
        self.current_extra_vectors = next_extra_vectors_list
        current_full_points = self._get_full_points(self.current_extra_vectors)
        next_nn_state = self._get_nn_state_representation(self.current_extra_vectors)

        current_info = {
            "points": current_full_points,
            "extra_vectors": self.current_extra_vectors,
            "is_idp": None,
            "h_star": None,
            "lc_metric": None,
            "is_unimodal": None,
        }  # Added is_unimodal
        reward_val = 0.0
        done_flag = False

        try:
            polytope_obj = PolytopeClass(current_full_points)
        except Exception as e:
            reward_val = -100.0
            done_flag = True
            current_info["error"] = str(e)
            return (
                next_nn_state,
                torch.tensor([reward_val], device=device),
                torch.tensor([done_flag], device=device, dtype=torch.bool),
                current_info,
            )

        is_idp = polytope_obj.is_idp
        current_info["is_idp"] = is_idp

        if not is_idp:
            reward_val = self.non_idp_penalty
        else:  # Is IDP
            try:
                h_star_raw = polytope_obj.h_star_vector
                h_star = list(h_star_raw)
                current_info["h_star"] = h_star
            except (AttributeError, TypeError):
                reward_val = -5.0
                h_star = None

            if h_star is not None:
                is_unimodal_val = get_unimodality_status(h_star)
                current_info["is_unimodal"] = is_unimodal_val
                lc_metric_val = calculate_normalized_min_log_concavity_metric(h_star)
                current_info["lc_metric"] = lc_metric_val

                if is_unimodal_val == False:  # Non-unimodal is the primary target
                    reward_val = self.reward_non_unimodal
                    done_flag = True  # Terminate on finding non-unimodal
                elif is_unimodal_val == True:  # Unimodal, check log-concavity
                    if lc_metric_val is None:  # LC or not applicable (e.g. too short)
                        reward_val = 1.0
                    elif lc_metric_val < 0:  # Non-LC but (still) Unimodal
                        reward_val = self.reward_non_lc_unimodal  # Episode continues
                    else:  # lc_metric_val >= 0 (IDP, Unimodal, and Log-Concave)
                        reward_val = 1.0 + (
                            self.K_lc_progress / (1.0 + lc_metric_val * 100)
                        )
                # else: is_unimodal_val is None (e.g. h_eff empty), treat as not a counterexample yet
                # This case means h_star was likely all zeros or very short, probably gets lc_metric_val=None too.
                # Default reward will be 0 unless lc_metric_val path above sets it.
                # If reward_val is still 0 here (e.g. unimodal=None, lc_metric=None), give base IDP
                if reward_val == 0.0 and (
                    is_unimodal_val is None or lc_metric_val is None
                ):
                    reward_val = (
                        0.5  # Small positive for IDP if other checks are inconclusive
                    )

        if reward_val > (self.non_idp_penalty - 5.0) and reward_val > -90:
            reward_val -= 0.1

        if self.current_step >= self.max_steps_per_episode and not done_flag:
            done_flag = True
        return (
            next_nn_state,
            torch.tensor([reward_val], device=device),
            torch.tensor([done_flag], device=device, dtype=torch.bool),
            current_info,
        )


# --- DQN Agent (same) ---
class DQNAgent:
    # ... (no changes needed here from previous version) ...
    def __init__(
        self,
        state_size,
        action_size,
        replay_memory_capacity=10000,
        batch_size=1,
        gamma=0.99,
        lr=1e-4,
        target_update_freq=10,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        hidden_factor_nn=3,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.policy_net = QNetwork(
            state_size, action_size, hidden_factor=hidden_factor_nn
        ).to(device)
        self.target_net = QNetwork(
            state_size, action_size, hidden_factor=hidden_factor_nn
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_capacity)

    def get_action(self, state_tensor):
        self.steps_done += 1
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        if random.random() > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state_tensor)).view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.action_size)]], device=device, dtype=torch.long
            )

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_q_values = torch.zeros(self.batch_size, device=device)
        non_final_mask = ~done_batch
        if non_final_mask.any():
            non_final_next_states = next_state_batch[non_final_mask]
            next_state_q_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1)[0].detach()
            )
        expected_q_values = (next_state_q_values * self.gamma) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath="dqn_policy_net.pth"):
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Policy network saved to {filepath}")

    def load_model(self, filepath="dqn_policy_net.pth"):
        try:
            self.policy_net.load_state_dict(torch.load(filepath, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
            print(f"Policy network loaded from {filepath}")
        except FileNotFoundError:
            print(f"No model found at {filepath}, starting fresh.")
        except Exception as e:
            print(f"Error loading model: {e}")


# --- Main Training Loop (UPDATED LOGIC/LOGGING) ---
def train_dqn(
    dimensions_to_try,
    num_extra_vectors_to_try,
    num_episodes,
    max_steps_per_episode=200,
    K_lc_progress_env=5.0,
    non_idp_penalty_env=-25.0,
    reward_non_unimodal_env=200.0,  # New param
    reward_non_lc_unimodal_env=50.0,  # New param
    replay_capacity=10000,
    batch_size_dqn=1,
    gamma_dqn=0.99,
    lr_dqn=1e-4,
    target_update_freq_dqn=10,
    eps_decay_dqn=10000,
    learning_starts=1,
    hidden_factor_agent=3,
):
    found_counterexamples = (
        []
    )  # Stores {..., "type": "non-unimodal" or "non-lc-unimodal"}
    overall_start_time = time.time()

    for d in dimensions_to_try:
        for n_extra in num_extra_vectors_to_try:
            print(f"\n--- DQN (Target: Non-Unimodal) Dim={d}, N_extra={n_extra} ---")

            env = PolytopeEnv(
                dimension=d,
                num_extra_vectors=n_extra,
                max_steps_per_episode=max_steps_per_episode,
                K_lc_progress=K_lc_progress_env,
                non_idp_penalty=non_idp_penalty_env,
                reward_non_unimodal=reward_non_unimodal_env,
                reward_non_lc_unimodal=reward_non_lc_unimodal_env,
            )

            agent = DQNAgent(
                state_size=env.state_size,
                action_size=env.action_space_size,
                replay_memory_capacity=replay_capacity,
                batch_size=batch_size_dqn,
                gamma=gamma_dqn,
                lr=lr_dqn,
                target_update_freq=target_update_freq_dqn,
                eps_decay=eps_decay_dqn,
                hidden_factor_nn=hidden_factor_agent,
            )

            model_filename = (
                f"dqn_nonuni_d{d}_ne{n_extra}_klc{K_lc_progress_env:.1f}_pen{abs(non_idp_penalty_env)}"
                f"_rnu{reward_non_unimodal_env:.0f}_rnlcu{reward_non_lc_unimodal_env:.0f}_b{batch_size_dqn}.pth"
            )
            agent.load_model(model_filename)

            episode_rewards_log = []
            total_steps_for_config = 0
            recent_lc_metrics = collections.deque(maxlen=200)
            losses_log = collections.deque(maxlen=200)

            for episode in range(num_episodes):
                ep_start_time = time.time()
                nn_state = env.reset()
                total_reward_episode = 0.0
                ep_lc_metrics = []
                idp_steps_current_episode = 0
                total_steps_current_episode = 0

                print(
                    f"  Ep {episode+1}/{num_episodes} (D{d},Ne{n_extra}) InitExtra: {env.current_extra_vectors}"
                )

                for step in range(max_steps_per_episode):
                    total_steps_for_config += 1
                    total_steps_current_episode += 1
                    action_tensor = agent.get_action(nn_state)
                    action_idx = action_tensor.item()

                    next_nn_state, reward_tensor, done_tensor, info = env.step(
                        action_idx
                    )

                    agent.store_transition(
                        nn_state,
                        action_tensor,
                        next_nn_state,
                        reward_tensor,
                        done_tensor,
                    )

                    nn_state = next_nn_state
                    reward_val = reward_tensor.item()
                    total_reward_episode += reward_val

                    is_idp_step = info.get("is_idp", False)
                    lc_metric_step = info.get("lc_metric")
                    h_star_step = info.get("h_star")
                    is_unimodal_step = info.get("is_unimodal")

                    if is_idp_step:
                        idp_steps_current_episode += 1
                        if (
                            lc_metric_step is not None
                            and lc_metric_step >= 0
                            and is_unimodal_step == True
                        ):  # Track only for LC & Unimodal cases
                            recent_lc_metrics.append(lc_metric_step)
                            ep_lc_metrics.append(lc_metric_step)

                    loss_val_step = None
                    if (
                        total_steps_for_config >= learning_starts
                        and len(agent.memory) >= agent.batch_size
                    ):
                        loss_val_step = agent.learn()
                        if loss_val_step is not None:
                            losses_log.append(loss_val_step)

                    lc_metric_str = (
                        f"{lc_metric_step:.3f}" if lc_metric_step is not None else "N/A"
                    )
                    h_star_str = str(h_star_step) if h_star_step is not None else "N/A"
                    loss_str = (
                        f"{loss_val_step:.4f}" if loss_val_step is not None else "N/L"
                    )
                    unimodal_str = (
                        str(is_unimodal_step) if is_unimodal_step is not None else "N/A"
                    )

                    print(
                        f"    D{d} Ne{n_extra} Ep{episode+1} St{step+1}: Act={action_idx}, Rew={reward_val:.2f}, IDP={is_idp_step}, "
                        f"Uni={unimodal_str}, LC_met={lc_metric_str}, h*={h_star_str}, Loss={loss_str}, Done={done_tensor.item()}"
                    )

                    if done_tensor.item():  # Episode ended
                        if "error" in info:
                            print(
                                f"    Episode ended due to Polytope error: {info['error']}"
                            )
                        elif is_idp_step and is_unimodal_step == False:
                            print(
                                f"    >>> SUCCESS! NON-UNIMODAL Counterexample Found! <<<"
                            )
                            print(
                                f"        Dim={d}, N_extra={n_extra}, Ep={episode+1}, Step={step+1}"
                            )
                            print(
                                f"        Uni={is_unimodal_step}, LC_Met={lc_metric_step:.4f if lc_metric_step is not None else 'N/A'}, h*={h_star_step}"
                            )
                            print(f"        Extra Vectors: {info.get('extra_vectors')}")
                            found_counterexamples.append(
                                {
                                    "type": "non-unimodal",
                                    "dimension": d,
                                    "num_extra_vectors": n_extra,
                                    "points": info.get("points"),
                                    "extra_vectors_only": info.get("extra_vectors"),
                                    "episode": episode + 1,
                                    "steps": step + 1,
                                    "h_star_vector": h_star_step,
                                    "lc_metric": lc_metric_step,
                                    "is_unimodal": is_unimodal_step,
                                }
                            )
                        # If it ended due to max_steps but found a non-LC (unimodal) one, we might want to log it differently
                        # but current logic only appends if "done" due to non-unimodal.
                        break

                current_episode_idp_rate = (
                    (idp_steps_current_episode / total_steps_current_episode)
                    if total_steps_current_episode > 0
                    else 0.0
                )
                episode_rewards_log.append(total_reward_episode)

                if episode > 0 and episode % agent.target_update_freq == 0:
                    agent.update_target_net()
                    # print(f"  Target network updated at end of Ep {episode+1}")

                ep_duration = time.time() - ep_start_time
                avg_recent_reward = (
                    np.mean(episode_rewards_log[-20:]) if episode_rewards_log else 0
                )
                current_epsilon = agent.eps_end + (
                    agent.eps_start - agent.eps_end
                ) * np.exp(-1.0 * agent.steps_done / agent.eps_decay)
                avg_recent_lc_metric = (
                    np.mean(list(recent_lc_metrics))
                    if recent_lc_metrics
                    else float("nan")
                )
                avg_ep_lc_metric = (
                    np.mean(ep_lc_metrics) if ep_lc_metrics else float("nan")
                )
                avg_loss_summary = (
                    np.mean(list(losses_log)) if losses_log else float("nan")
                )

                print(
                    f"  Ep {episode+1} END: TotalRew={total_reward_episode:.2f} (AvgLR20={avg_recent_reward:.2f}), Eps={current_epsilon:.3f}, "
                    f"Loss={avg_loss_summary:.4f}, AvgLC_Met(ep)={avg_ep_lc_metric:.3f} (glob)={avg_recent_lc_metric:.3f}, "
                    f"IDP_Rate(curr_ep)={current_episode_idp_rate:.2%}, StepsTaken={step+1}, Time={ep_duration:.2f}s\n"
                )

            agent.save_model(model_filename)

    total_duration = time.time() - overall_start_time
    print(f"\n--- Total Training Duration: {total_duration:.2f} seconds ---")

    if found_counterexamples:
        print("\n--- DQN (Target: Non-Unimodal) Found Counterexamples ---")
        for i, ex in enumerate(found_counterexamples):
            lc_metric_val = ex.get("lc_metric")
            lc_metric_str = (
                f"{lc_metric_val:.4f}"
                if isinstance(lc_metric_val, (int, float))
                else str(lc_metric_val)
            )
            print(f"\n  Counterexample {i+1}: Type: {ex.get('type', 'N/A')}")
            print(
                f"    Dim={ex['dimension']}, N_extra={ex['num_extra_vectors']}, Ep={ex['episode']}, Steps={ex['steps']}"
            )
            print(f"    Extra Vecs: {ex['extra_vectors_only']}")
            print(f"    h* vector: {ex['h_star_vector']}")
            print(
                f"    Unimodal: {ex.get('is_unimodal')}, Norm_LC_Metric: {lc_metric_str}"
            )
        counterexamples_filename = (
            f"dqn_nonuni_found_counterexamples_klc{K_lc_progress_env:.1f}_pen{abs(non_idp_penalty_env)}"
            f"_rnu{reward_non_unimodal_env:.0f}_rnlcu{reward_non_lc_unimodal_env:.0f}_b{batch_size_dqn}_final.pkl"
        )
        try:
            with open(counterexamples_filename, "wb") as f:
                pickle.dump(found_counterexamples, f)
            print(
                f"DQN (Target: Non-Unimodal) Counterexamples saved to {counterexamples_filename}"
            )
        except Exception as e:
            print(f"Error saving counterexamples: {e}")
    else:
        print("\n--- DQN (Target: Non-Unimodal): No counterexamples found. ---")
    return found_counterexamples


if __name__ == "__main__":
    dims_dqn = [30]
    num_extras_dqn = [2]
    episodes_dqn = 50
    steps_per_episode_dqn = 30
    K_lc_param_dqn = 3.0  # Might need to be lower if non-LC reward is high
    non_idp_penalty_main = -25.0
    # New rewards for unimodality objective
    reward_non_unimodal_main = 200.0
    reward_non_lc_unimodal_main = 50.0

    replay_mem_cap = 2000
    dqn_batch_size = 1
    dqn_gamma = 0.99
    dqn_lr = 3e-5  # Further reduced LR for stability
    dqn_target_update = 5
    total_test_steps = (
        episodes_dqn * steps_per_episode_dqn * len(dims_dqn) * len(num_extras_dqn)
    )
    dqn_eps_decay = max(200, total_test_steps / 3)  # Slower decay for more exploration
    dqn_learning_starts = 1
    dqn_hidden_factor = 4

    print(
        f"Starting DQN training (Target: Non-Unimodal): Non-IDP Penalty={non_idp_penalty_main}, Batch Size={dqn_batch_size}, LR={dqn_lr}"
    )
    print(
        f"Reward Non-Unimodal: {reward_non_unimodal_main}, Reward Non-LC (but Unimodal): {reward_non_lc_unimodal_main}"
    )
    print(f"Epsilon will decay over approx. {dqn_eps_decay} steps.")

    try:  # Test Polytope class
        test_dim, test_n_extra = dims_dqn[0], num_extras_dqn[0]
        test_basis = [
            [1 if i == j else 0 for i in range(test_dim)] for j in range(test_dim)
        ]
        test_origin = [0] * test_dim
        test_extra = [
            [random.randint(0, 1) for _ in range(test_dim)] for _ in range(test_n_extra)
        ]
        test_points = [test_origin] + test_basis + test_extra
        if not PolytopeClass:
            raise NameError("PolytopeClass is not defined.")
        p_test = PolytopeClass(test_points)
        _ = p_test.is_idp
        h_test = p_test.h_star_vector
        _ = list(h_test)
        # Test unimodality function
        test_h = [1, 2, 3, 2, 1]
        print(f"Test h* {test_h} unimodal: {get_unimodality_status(test_h)}")
        test_h_non = [1, 3, 2, 4, 1]
        print(f"Test h* {test_h_non} unimodal: {get_unimodality_status(test_h_non)}")
        test_h_zero_mid = [1, 2, 0, 2, 1]
        print(
            f"Test h* {test_h_zero_mid} unimodal: {get_unimodality_status(test_h_zero_mid)}"
        )
        print("Test Polytope object created and attributes accessed successfully.")
    except Exception as e:
        print(f"ERROR testing: {e}")
        exit()

    found_examples_dqn = train_dqn(
        dimensions_to_try=dims_dqn,
        num_extra_vectors_to_try=num_extras_dqn,
        num_episodes=episodes_dqn,
        max_steps_per_episode=steps_per_episode_dqn,
        K_lc_progress_env=K_lc_param_dqn,
        non_idp_penalty_env=non_idp_penalty_main,
        reward_non_unimodal_env=reward_non_unimodal_main,
        reward_non_lc_unimodal_env=reward_non_lc_unimodal_main,
        replay_capacity=replay_mem_cap,
        batch_size_dqn=dqn_batch_size,
        gamma_dqn=dqn_gamma,
        lr_dqn=dqn_lr,
        target_update_freq_dqn=dqn_target_update,
        eps_decay_dqn=dqn_eps_decay,
        learning_starts=dqn_learning_starts,
        hidden_factor_agent=dqn_hidden_factor,
    )

    if found_examples_dqn:
        print("\n--- Summary of DQN (Target: Non-Unimodal) Found Counterexamples ---")
        for i, ex in enumerate(found_examples_dqn):
            lc_metric_val = ex.get("lc_metric")
            lc_metric_str = (
                f"{lc_metric_val:.4f}"
                if isinstance(lc_metric_val, (int, float))
                else str(lc_metric_val)
            )
            print(f"\n  Counterexample {i+1}: Type: {ex.get('type', 'N/A')}")
            print(
                f"    Dim={ex['dimension']}, N_extra={ex['num_extra_vectors']}, Ep={ex['episode']}, Steps={ex['steps']}"
            )
            print(f"    Extra Vecs: {ex['extra_vectors_only']}")
            print(f"    h* vector: {ex['h_star_vector']}")
            print(
                f"    Unimodal: {ex.get('is_unimodal')}, Norm_LC_Metric: {lc_metric_str}"
            )
    else:
        print(
            "\n--- DQN (Target: Non-Unimodal): No counterexamples were found in this run. ---"
        )
