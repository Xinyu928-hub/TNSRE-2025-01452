import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from MPC.BCI_FT import env_cars

# Actor-Critic neural network used in PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128], activation=F.relu):
        super(ActorCritic, self).__init__()
        layers = []
        input_dim = state_dim

        # Create hidden layers
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_dim, hs))  # Linear layer
            layers.append(nn.ReLU())  # Activation
            input_dim = hs
        self.shared = nn.Sequential(*layers)  # Shared base network
        self.actor = nn.Linear(input_dim, action_dim)  # Policy head (for actions)
        self.critic = nn.Linear(input_dim, 1)  # Value head (for state value)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)  # Return logits and state value

    def act(self, state):
        # Get action from policy
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        dist = torch.distributions.Categorical(probs)  # Discrete distribution
        action = dist.sample()  # Sample an action
        return action.item(), dist.log_prob(action), dist.entropy()  # Return action, log prob, and entropy

# PPO Trainer class
class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, device, clip_param=0.2, ppo_epochs=4, batch_size=128, gamma=0.99, lam=0.95, lr=3e-4):
        self.model = ActorCritic(state_dim, action_dim).to(device)  # Actor-Critic model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Optimizer
        self.clip_param = clip_param # Clipping range for PPO
        self.ppo_epochs = ppo_epochs # Number of PPO updates per batch
        self.batch_size = batch_size # Mini-batch size
        self.gamma = gamma # Discount factor
        self.lam = lam # GAE lambda
        self.device = device # CPU/GPU device

    def compute_returns(self, rewards, masks, values, next_value):
        # Compute returns and advantages using Generalized Advantage Estimation (GAE)
        returns, advs = [], []
        gae = 0
        values = values + [next_value]  # Append next value to value list
        for step in reversed(range(len(rewards))):
            # TD error
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advs

    def train_one_batch(self, states, actions, log_probs_old, returns, advantages):
        # Convert data to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        log_probs_old = torch.stack(log_probs_old).detach().to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(self.ppo_epochs):
            # Get new logits and values
            logits, values = self.model(states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs_new = dist.log_prob(actions) # New log probabilities
            entropy = dist.entropy().mean() # Policy entropy (encourages exploration)

            # PPO objective: ratio of new/old probabilities
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # Clipped actor loss

            critic_loss = F.mse_loss(values.squeeze(), returns) # Value function loss

            # Total loss = actor + critic - entropy (for exploration)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Utility: convert state (which may be list or tuple) to numpy array
def state_to_ndarray(state):
    return np.array([state.x, state.y, state.yaw, state.v])


if __name__ == "__main__":
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    num_episodes = 350
    tqdm_num = 10

    env = env_cars()
    env.rho = 0.1
    env.MAXTIME = 40
    env.bdelta = 5
    env.adelta = 0.1

    ppo_trainer = PPOTrainer(env, env.obs_space, env.action_space, device=device)

    all_episode_rewards = []
    episodes_per_block = num_episodes // tqdm_num

    for block_idx in range(tqdm_num):
        with tqdm(total=episodes_per_block, desc=f'Iteration {block_idx}') as pbar:
            for episode_idx in range(episodes_per_block):
                state = env.reset()
                state_array = state_to_ndarray(state)
                done = False
                episode_reward = 0

                # Storage for trajectories
                states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []

                while not done:
                    # Convert state to tensor
                    state_tensor = torch.tensor(state_array, dtype=torch.float32).to(device)

                    # Select action using current policy (no gradient tracking)
                    with torch.no_grad():
                        action, log_prob, _ = ppo_trainer.model.act(state_tensor)
                        _, value = ppo_trainer.model(state_tensor)

                    # Take action in the environment
                    is_done, next_state, reward, done = env.step(action, state)
                    next_state_array = state_to_ndarray(next_state)

                    # Store trajectory
                    states.append(state_tensor)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    masks.append(1 - float(is_done))
                    values.append(value.item())

                    # Accumulate reward
                    episode_reward += reward

                    # Update current state
                    state_array = next_state_array
                    state = next_state

                # Estimate value for the next (terminal) state
                next_state_tensor = torch.tensor(next_state_array, dtype=torch.float32).to(device)
                with torch.no_grad():
                    _, next_value = ppo_trainer.model(next_state_tensor)

                # Compute GAE returns and advantages
                returns, advantages = ppo_trainer.compute_returns(rewards, masks, values, next_value.item())

                # Update policy using collected batch
                ppo_trainer.train_one_batch(states, actions, log_probs, returns, advantages)

                # Update tqdm progress bar
                all_episode_rewards.append(episode_reward)
                pbar.set_postfix({'episode': block_idx * episodes_per_block + episode_idx + 1, 'return': f'{episode_reward:.3f}'})
                pbar.update(1)
