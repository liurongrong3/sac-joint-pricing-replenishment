import os
import random
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config_loader import DEFAULT_CONFIG_PATH, get_paths, get_section, load_config, model_path
from sac_agent import ReplayBuffer, SACAgent
from shopee_env import ShopeeSupplyChainEnv


def set_global_seed(seed: int):
    """固定 Python / NumPy / PyTorch 随机种子，便于复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 1. 课程学习管理器 (Curriculum Manager)
# =====================================================================
class CurriculumManager:

    def __init__(self, difficulties=None, thresholds=None, window_size=10):
        self.difficulties = (
            difficulties if difficulties else ["easy", "medium", "hard"]
        )
        self.thresholds = (
            thresholds if thresholds else [35000.0, 75000.0]
        )  # 与 config.json training.curriculum_thresholds 对齐
        self.current_idx = 0
        self.window_size = window_size
        self.reward_window = deque(maxlen=window_size)

    @property
    def current_difficulty(self):
        return self.difficulties[self.current_idx]

    @property
    def current_level(self):
        return self.current_idx

    def add_reward(self, reward):
        self.reward_window.append(reward)

    def check_upgrade(self):
        if len(self.reward_window) < self.window_size:
            return False, self.current_difficulty

        avg_reward = np.mean(self.reward_window)

        if self.current_idx < len(self.thresholds):
            target_threshold = self.thresholds[self.current_idx]
            if avg_reward >= target_threshold:
                self.current_idx += 1
                self.reward_window.clear()
                print(
                    f"\n[🎉 课程升级] 最近 {self.window_size} 回合平均回报 RM {avg_reward:,.2f} "
                    f"超越阈值 RM {target_threshold:,.2f}！"
                )
                print(
                    f"正切换至新难度级别: ---> 【{self.current_difficulty.upper()}】 <---"
                )
                return True, self.current_difficulty

        return False, self.current_difficulty




# =====================================================================
# 2. 带有课程学习与 TensorBoard 监控的训练主循环
# =====================================================================
def train_sac_curriculum(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    sac_cfg = get_section(config, "sac_agent")
    train_cfg = get_section(config, "training")
    paths = get_paths(config, config_path, mkdir=True)

    seed = int(train_cfg.get("seed", 42))
    set_global_seed(seed)
    print(f"随机种子: {seed}")

    run_name = f"{paths['tensorboard_run_prefix']}_{int(time.time())}"
    log_dir = os.path.join(paths["tensorboard_dir"], run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    models_dir = paths["models_dir"]
    best_reward = -float("inf")

    log_file_path = paths["train_log"]
    log_f = open(log_file_path, "w", encoding="utf-8")

    curriculum = CurriculumManager(
        difficulties=train_cfg.get("curriculum_difficulties", ["easy", "medium", "hard"]),
        thresholds=train_cfg.get("curriculum_thresholds", [35000.0, 75000.0]),
        window_size=train_cfg.get("curriculum_window_size", 10),
    )

    env = ShopeeSupplyChainEnv(config_path=config_path, difficulty=curriculum.current_difficulty)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, **sac_cfg)
    memory = ReplayBuffer(capacity=train_cfg.get("replay_buffer_capacity", 200000))

    max_episodes = train_cfg.get("max_episodes", 200)
    batch_size = train_cfg.get("batch_size", 128)
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    global_step = 0

    print("=" * 80)
    print(f"🚀 开始课程学习 SAC 训练 | 初始难度: {curriculum.current_difficulty}")
    print("=" * 80)

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        critic_losses, actor_losses, alpha_values = [], [], []

        ep_financials = {
            "gross_profit": 0.0,
            "storage_cost": 0.0,
            "stockout_cost": 0.0,
            "order_cost": 0.0,
            "net_profit": 0.0,
        }

        for step in range(env.max_steps):
            global_step += 1

            if global_step < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = float(terminated)

            if isinstance(info, dict):
                fin_source = info.get("financials", info)
                for key in ep_financials.keys():
                    if key in fin_source:
                        ep_financials[key] += float(fin_source[key])

            # 修复点 1：写入 Replay Buffer 时必须使用缩放后的 scaled_reward，防止 Q 值爆炸
            scaled_reward = reward * agent.reward_scale
            memory.push(state, action, scaled_reward, next_state, done)
            
            state = next_state
            ep_reward += reward  # 真实 P&L 统计（不缩放）

            if global_step >= warmup_steps:
                losses = agent.update_parameters(memory, batch_size)
                if losses:
                    critic_losses.append(losses["critic_loss"])
                    actor_losses.append(losses["actor_loss"])
                    alpha_values.append(losses["alpha"])

            if terminated or truncated:
                break

        curriculum.add_reward(ep_reward)
        upgraded, new_diff = curriculum.check_upgrade()

        avg_c_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_a_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else agent.alpha.item()
        win_avg_reward = (
            np.mean(curriculum.reward_window)
            if curriculum.reward_window
            else ep_reward
        )

        # -----------------------------------------------------------------
        # 保存最佳权重与模型持久化
        # -----------------------------------------------------------------
        if win_avg_reward > best_reward:
            best_reward = win_avg_reward
            best_model_path = model_path(
                paths, difficulty=curriculum.current_difficulty
            )
            torch.save(agent.actor.state_dict(), best_model_path)
            
            save_msg = f"🏆 [{curriculum.current_difficulty.upper()}] 突破最高窗口平均奖励 ({best_reward:,.2f})！已保存最佳模型。"
            print(save_msg)
            print(save_msg, file=log_f, flush=True)

        # TensorBoard 监控
        writer.add_scalar("Train/Episode_Reward", ep_reward, ep)
        writer.add_scalar("Train/Avg_Window_Reward", win_avg_reward, ep)
        writer.add_scalar("Curriculum/Level", curriculum.current_level, ep)
        writer.add_scalar("Loss/Critic_Loss", avg_c_loss, ep)
        writer.add_scalar("Loss/Actor_Loss", avg_a_loss, ep)
        writer.add_scalar("Loss/Alpha", avg_alpha, ep)

        writer.add_scalar("Financials/Storage_Cost", ep_financials["storage_cost"], ep)
        writer.add_scalar("Financials/Stockout_Cost", ep_financials["stockout_cost"], ep)
        writer.add_scalar("Financials/Order_Cost", ep_financials["order_cost"], ep)
        writer.add_scalar("Financials/Gross_Profit", ep_financials["gross_profit"], ep)
        writer.add_scalar("Financials/Net_Profit", ep_financials["net_profit"], ep)

        # 修复点 3：同时输出到控制台 Terminal 与日志文件
        log_msg = (
            f"Ep {ep:03d}/{max_episodes:03d} | 难度: {curriculum.current_difficulty:<6} | "
            f"P&L: RM {ep_reward:>10,.2f} | 仓储费: RM {ep_financials['storage_cost']:>8,.2f} | "
            f"缺货罚金: RM {ep_financials['stockout_cost']:>8,.2f} | Alpha: {avg_alpha:.4f}"
        )
        # 不输出到终端
        # print(log_msg)
        print(log_msg, file=log_f, flush=True)

        # 修复点 2：升难时重置 best_reward 并可选清空经验池，防止跨难度污染
        if upgraded:
            upgrade_msg = f"\n🎉 恭喜！触发课程升级，进入【{new_diff.upper()}】难度阶段！"
            print(upgrade_msg)
            print(upgrade_msg, file=log_f, flush=True)
            
            env = ShopeeSupplyChainEnv(config_path=config_path, difficulty=new_diff)
            best_reward = -float("inf")  # 重置最高奖励门槛
            memory.buffer.clear()        # 清空旧难度经验池，消除 Distribution Shift

    final_model_path = model_path(paths, final=True)
    torch.save(agent.actor.state_dict(), final_model_path)
    
    log_f.close()
    writer.close()
    print("=" * 80)
    print(f"训练完成！模型已保存至 {models_dir}，TensorBoard 日志: {log_dir}")

if __name__ == "__main__":
    train_sac_curriculum()




# 课程学习与监控架构课程学习 (Curriculum Learning)：难度阶段：easy  medium  hard。
# 晋级机制：维护滑动窗口（如最近 10 回合）的平均 Episode 利润/回报。当滑动平均回报达到预设阈值且达到最少评估回合数时，自动提升难度并重置/重构环境。
# TensorBoard 监控面板：
# 性能指标：Episode Total Reward / P&L、滑动平均收益。
# 算法损失与参数：Critic Loss、Actor Loss、自适应熵系数alpha 。
# 课程状态：当前难度等级（0: Easy, 1: Medium, 2: Hard）。


# 训练开始后，可以打开终端输入以下命令启动 TensorBoard 界面：
# tensorboard --logdir=runs

# 在浏览器访问 http://localhost:6006，你将直观看到：
# Train/Episode_Reward 与 Train/Avg_Window_Reward：判断策略收益的增长与稳定性。
# Curriculum/Level：阶梯状折线图（0  1  2），直观反映智能体何时突破困难瓶颈。
# Loss/Alpha：观测最大熵系数 alpha 在训练过程中的收敛与自适应调整过程。
