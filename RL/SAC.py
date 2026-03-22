import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import deque, namedtuple
from MPC.BCI_FT import env_cars

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)

# Actor network for discrete actions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128]):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

# Critic network for Q-function
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128]):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], action_dim)
        )

    def forward(self, x):
        return self.net(x)  # Q-values for all actions

class SACAgent:
    def __init__(self, state_dim, action_dim, device, gamma=0.99, alpha=0.2, lr=3e-4, tau=0.005):
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        batch = replay_buffer.sample(batch_size)
        state = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action = torch.tensor(batch.action, dtype=torch.int64).to(self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(batch.done, dtype=torch.float32).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_state)
            q1_next = self.target_critic1(next_state)
            q2_next = self.target_critic2(next_state)
            min_q_next = torch.min(q1_next, q2_next)

            entropy_term = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward + (1 - done) * self.gamma * entropy_term

        # Critic update
        q1 = self.critic1(state).gather(1, action)
        q2 = self.critic2(state).gather(1, action)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # Actor update
        probs, log_probs = self.actor(state)
        with torch.no_grad():
            q1_pi = self.critic1(state)
            q2_pi = self.critic2(state)
            min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Target network update
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# State preprocessing
def state_to_ndarray(state):
    return np.array([state.x, state.y, state.yaw, state.v])

if __name__ == "__main__":
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_episodes = 350
    tqdm_num = 10
    batch_size = 128

    env = env_cars()
    env.rho = 0.1
    env.MAXTIME = 40
    env.bdelta = 5
    env.adelta = 0.1

    agent = SACAgent(env.obs_space, env.action_space, device)
    replay_buffer = ReplayBuffer(capacity=10000)

    all_episode_rewards = []
    episodes_per_block = num_episodes // tqdm_num

    for block_idx in range(tqdm_num):
        with tqdm(total=episodes_per_block, desc=f"Iteration {block_idx}") as pbar:
            for episode_idx in range(episodes_per_block):
                state = env.reset()
                state_array = state_to_ndarray(state)
                episode_reward = 0
                done = False

                while not done:
                    action = agent.select_action(state_array)
                    is_done, next_state, reward, done = env.step(action, state)
                    next_state_array = state_to_ndarray(next_state)

                    replay_buffer.push(state_array, action, reward, next_state_array, is_done)
                    agent.update(replay_buffer, batch_size)

                    state = next_state
                    state_array = next_state_array
                    episode_reward += reward

                all_episode_rewards.append(episode_reward)
                pbar.set_postfix({'episode': block_idx * episodes_per_block + episode_idx + 1,
                                  'return': f'{episode_reward:.3f}'})
                pbar.update(1)
