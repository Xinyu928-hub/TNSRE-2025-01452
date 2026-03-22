import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from MPC.BCI_FT import env_cars

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

seed = 10
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#mode = "DUE" #DUE or DDQN
mode = "DDQN"
num_episodes = 350
target_update = 250*2
tqdm_num = 10
save_train_model = 10
batch_size = 128
gamma = 0.99
epsilon_start = 1
epsilon_final = 0.05
epsilon_decay = 350*60
beta_start = 0.4
beta_final = 1
beta_decay = 350*140
buffer_size = 5000
env = env_cars()
env.rho = 0.1
env.MAXTIME = 40
env.bdelta = 5
env.adelta = 0.1

dir_name =  '_mode_'+ str(mode) + '_seed_' + str(seed)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if not os.path.exists(dir_name + "/save_model"):
    os.mkdir(dir_name + "/save_model")

def state_to_ndarray(state):
    return np.array([state.x, state.y, state.yaw, state.v])

class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, states, actions, rewards, next_states, dones):
        assert len(states) == len(next_states) == len(actions) == len(rewards) == len(dones), "长度不匹配"
        #assert state.ndim == next_state.ndim
        for i in range(len(states)):
            state = np.expand_dims(states[i], 0)
            next_state = np.expand_dims(next_states[i], 0)

            max_prio = self.priorities.max() if self.buffer else 1.0

            if len(self.buffer) < self.capacity:
                self.buffer.append((state, actions[i], rewards[i], next_state, dones[i]))
            else:
                self.buffer[self.pos] = (state, actions[i], rewards[i], next_state, dones[i])

            self.priorities[self.pos] = max_prio
            #self.priorities[self.pos] = td_error_based_prio
            self.pos = (self.pos + 1) % self.capacity


    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]
        return states, actions, rewards, next_states, dones, indices, weights


    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes=[128], activation=F.relu):
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions
        self.activation = activation

        # Shared hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = num_inputs
        for next_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_size, next_size))
            in_size = next_size

        # Dueling branches: value and advantage
        self.value_stream     = nn.Linear(in_size, 1)
        self.advantage_stream = nn.Linear(in_size, num_actions)

    def forward(self, x):
        single_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True

        x = x.to(device)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        value     = self.value_stream(x)               # (batch, 1)
        advantage = self.advantage_stream(x)           # (batch, num_actions)

        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        # 如果原本是单个状态，输出也 squeeze 回 (num_actions,)
        if single_input:
            q_values = q_values.squeeze(0)

        return q_values

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes=[128], activation=F.relu):
        super(DQN, self).__init__()
        self.activation = activation

        # Shared hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = num_inputs
        for size in hidden_sizes:
            fc = nn.Linear(in_size, size)
            in_size = size
            self.hidden_layers.append(fc)

        # Output layer for action values Q(s, a)
        self.output_layer = nn.Linear(in_size, num_actions)

    def forward(self, x):
        x = x.to(device)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        q_values = self.output_layer(x)
        return q_values

def update_target(current_model, target_model):
    # Copy parameters from current to target model
    target_model.load_state_dict(current_model.state_dict())

def compute_td_loss(batch_size, beta):
    # Sample from replay buffer with PER
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)

    # Convert to tensors
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # Compute Q(s,a) from current model
    q_values = current_model(states)  # shape: [batch_size, num_actions]
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Q-value of taken actions

    # Use Double DQN: action from current, value from target
    next_q_values_current = current_model(next_states)
    next_actions = next_q_values_current.max(1)[1].unsqueeze(1)  # index of best actions

    next_q_values_target = target_model(next_states)
    next_q_value = next_q_values_target.gather(1, next_actions).squeeze(1)

    # Bellman target
    expected_q_value = rewards + gamma * (1 - dones) * next_q_value

    # Compute TD error
    td_errors = q_value - expected_q_value.detach()
    loss = (td_errors.pow(2) * weights).mean()

    # Update priorities
    priorities = td_errors.abs().detach() + 1e-5
    replay_buffer.update_priorities(indices, priorities.cpu().numpy())

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

if __name__ == "__main__":
    if mode == "DDQN":
        current_model = DQN(env.obs_space, env.action_space, hidden_sizes=[128], activation=F.relu).to(device)
        target_model = DQN(env.obs_space, env.action_space, hidden_sizes=[128], activation=F.relu).to(device)
    elif mode == "DUE":
        current_model = DuelingDQN(env.obs_space, env.action_space, hidden_sizes=[128], activation=F.relu).to(device)
        target_model = DuelingDQN(env.obs_space, env.action_space, hidden_sizes=[128], activation=F.relu).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create optimizer and experience replay buffer
    optimizer = optim.Adam(current_model.parameters())
    replay_buffer = NaivePrioritizedBuffer(buffer_size)

    all_episode_rewards = []  # Store total reward for each episode
    losses = []               # Store training loss
    step_counter = 0

    episodes_per_block = num_episodes // tqdm_num # Episodes per tqdm display block

    for block_idx in range(tqdm_num):
        with tqdm(total=episodes_per_block, desc=f'Iteration {block_idx}') as pbar:
            for episode_idx in range(episodes_per_block):
                global_episode = block_idx * episodes_per_block + episode_idx + 1

                # Save model periodically
                if global_episode % save_train_model == 0:
                    torch.save(current_model.state_dict(), f"{dir_name}/save_model/w_{global_episode}.pt")

                state = env.reset()
                state_array = state_to_ndarray(state)

                episode_reward = 0
                done = False

                # Trajectory placeholders
                states, actions, rewards, next_states, dones = [], [], [], [], []

                while not done:
                    step_counter += 1

                    # Update state if not the first step
                    if step_counter > 1:
                        state = next_state
                        state_array = next_state_array

                    # Linear decay epsilon-greedy strategy
                    epsilon = np.interp(step_counter, [0, epsilon_decay], [epsilon_start, epsilon_final])
                    state_tensor = torch.from_numpy(state_array).float().to(device)
                    q_value = current_model(state_tensor)

                    if random.random() > epsilon:
                        action = q_value.argmax().item()
                    else:
                        action = random.randrange(env.action_space)

                    # Store trajectory elements
                    actions.append(action)
                    states.append(state_array)

                    # Environment interaction
                    is_done, next_state, reward, done = env.step(action, state)
                    next_state_array = state_to_ndarray(next_state)

                    next_states.append(next_state_array)
                    rewards.append(reward)
                    dones.append(is_done)

                    if len(replay_buffer) > batch_size:
                        beta = np.interp(step_counter, [0, beta_decay], [beta_start, beta_final])
                        loss = compute_td_loss(batch_size, beta)
                        losses.append(loss.item())

                    # Target network soft/hard update
                    if step_counter % target_update == 0:
                        update_target(current_model, target_model)

                episode_reward = sum(rewards)
                all_episode_rewards.append(episode_reward)

                # Add entire episode to replay buffer (multi-step learning support)
                replay_buffer.push(states, actions, rewards, next_states, dones)

                pbar.set_postfix({
                    'episode': global_episode,
                    'return': f'{episode_reward:.3f}'
                })
                pbar.update(1)
