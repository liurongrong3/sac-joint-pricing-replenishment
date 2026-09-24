import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# 导入供应链环境
from shopee_env import ShopeeSupplyChainEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. 经验回放池 (Replay Buffer)
# =====================================================================
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# =====================================================================
# 2. Twin Q-Network (Double Critic)
# =====================================================================
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa), self.q2_net(sa)


# =====================================================================
# 3. Gaussian Actor (策略网络)
# =====================================================================
class Actor(nn.Module):

    def __init__(
        self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2
    ):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


# =====================================================================
# 4. SAC Agent 算法核心主体
# =====================================================================
class SACAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        hidden_dim=256,
        reward_scale=1e-4,
        log_alpha_min=-5.0,
        log_alpha_max=2.0,
        grad_clip_max_norm=1.0,
        log_std_min=-20.0,
        log_std_max=2.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.log_alpha_min = log_alpha_min
        self.log_alpha_max = log_alpha_max
        self.grad_clip_max_norm = grad_clip_max_norm

        # 1. Critic 网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # 2. Actor 网络
        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)

        # 3. 自动调整熵系数 alpha
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        if len(memory) < batch_size:
            return {}

        state, action, reward, next_state, done = memory.sample(batch_size)

        # -----------------------------------------------------------------
        # Step 1: 更新 Critic
        # -----------------------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next_target, q2_next_target = self.critic_target(
                next_state, next_action
            )
            min_q_next_target = (
                torch.min(q1_next_target, q2_next_target)
                - self.alpha * next_log_prob
            )
            next_q_value = reward + (1 - done) * self.gamma * min_q_next_target

        q1, q2 = self.critic(state, action)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip_max_norm)
        self.critic_optimizer.step()

        # -----------------------------------------------------------------
        # Step 2: 更新 Actor
        # -----------------------------------------------------------------
        pi, log_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_max_norm)
        self.actor_optimizer.step()

        # -----------------------------------------------------------------
        # Step 3: 更新 Alpha 并强行截断 (防止 Alpha 爆炸)
        # -----------------------------------------------------------------
        alpha_loss = -(
            self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.log_alpha.data.clamp_(min=self.log_alpha_min, max=self.log_alpha_max)

        # -----------------------------------------------------------------
        # Step 4: 软更新 Target Critic
        # -----------------------------------------------------------------
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }


# # =====================================================================
# # 🧪 训练主循环
# # =====================================================================
# if __name__ == "__main__":
    # print("=" * 80)
    # print("🚀 初始化 SAC 智能体并测试训练流程...")
    # print("=" * 80)

    # env = ShopeeSupplyChainEnv(difficulty="easy")
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]

    # agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    # memory = ReplayBuffer(capacity=100000)

    # batch_size = 128
    # episodes = 20
    # warmup_steps = 500
    # total_steps = 0

    # for ep in range(1, episodes + 1):
    #     state, _ = env.reset()
    #     ep_reward = 0.0
    #     critic_losses, actor_losses = [], []

    #     for step in range(env.max_steps):
    #         total_steps += 1

    #         if total_steps < warmup_steps:
    #             action = env.action_space.sample()
    #         else:
    #             action = agent.select_action(state, evaluate=False)

    #         next_state, reward, terminated, truncated, info = env.step(action)
    #         done = float(terminated)

    #         # 核心修改：写入 Buffer 时做 Reward Scaling (乘 1e-4)
    #         scaled_reward = reward * agent.reward_scale
    #         memory.push(state, action, scaled_reward, next_state, done)

    #         state = next_state
    #         ep_reward += reward  # 记录真实的未经缩放的绝对 P&L 利润

    #         if total_steps >= warmup_steps:
    #             losses = agent.update_parameters(memory, batch_size)
    #             if losses:
    #                 critic_losses.append(losses["critic_loss"])
    #                 actor_losses.append(losses["actor_loss"])

    #         if terminated or truncated:
    #             break

    #     avg_c_loss = np.mean(critic_losses) if critic_losses else 0.0
    #     avg_a_loss = np.mean(actor_losses) if actor_losses else 0.0

    #     print(
    #         f"Episode {ep:02d}/{episodes:02d} | 实际总利润 (P&L): RM {ep_reward:>10,.2f} | "
    #         f"Critic Loss: {avg_c_loss:.4f} | Actor Loss: {avg_a_loss:.4f} | Alpha: {agent.alpha.item():.4f}"
    #     )

    # print("\n✅ SAC 修复模型搭建校验完成！")