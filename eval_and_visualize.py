import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config_loader import (
    DEFAULT_CONFIG_PATH,
    get_paths,
    get_section,
    load_config,
    model_path,
)
from sac_agent import SACAgent
from shopee_env import ShopeeSupplyChainEnv

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def load_trained_agent(model_path, state_dim, action_dim, sac_cfg):
    """加载预训练好的 SAC Actor 模型。"""
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, **sac_cfg)
    if os.path.exists(model_path):
        agent.actor.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.actor.eval()
        print(f"成功加载模型权重: {model_path}")
    else:
        print(f"未找到权重文件 {model_path}，将使用随机初始化的 Actor 运行测试。")
    return agent


def run_evaluation_episode(env, agent, seed=None):
    """运行单 Episode 评估，收集逐时步决策与环境指标。"""
    state, _ = env.reset(seed=seed)
    history = {
        "step": [],
        "inventory": [],
        "order_qty": [],
        "selling_price": [],
        "demand": [],
        "sales_qty": [],
        "step_reward": [],
        "cum_net_profit": [],
        "storage_cost": [],
        "stockout_cost": [],
        "order_cost": [],
    }

    cum_profit = 0.0

    for step in range(env.max_steps):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, info = env.step(action)

        cum_profit += info.get("net_profit", reward)

        history["step"].append(step + 1)
        history["inventory"].append(info.get("ending_inventory", state[0]))
        history["order_qty"].append(info.get("order_qty", action[0]))
        history["selling_price"].append(info.get("price", action[1] if len(action) > 1 else 0))
        history["demand"].append(info.get("actual_demand", 0.0))
        history["sales_qty"].append(info.get("actual_sales", 0.0))
        history["step_reward"].append(reward)
        history["cum_net_profit"].append(cum_profit)
        history["storage_cost"].append(info.get("storage_cost", 0.0))
        history["stockout_cost"].append(info.get("stockout_cost", 0.0))
        history["order_cost"].append(info.get("order_cost", 0.0))

        state = next_state
        if terminated or truncated:
            break

    return pd.DataFrame(history)


def plot_decision_curves(df, save_path):
    """绘制 3 合 1 供应链决策与财务分析图。"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    steps = df["step"]

    ax1 = axes[0]
    ax1.plot(steps, df["inventory"], color="#1f77b4", linewidth=2, label="期末库存 (Inventory)")
    ax1.plot(
        steps, df["demand"], color="#d62728", linestyle="--", alpha=0.8, label="市场需求 (Demand)"
    )
    ax1.bar(
        steps, df["order_qty"], color="#2ca02c", alpha=0.4, width=0.8, label="补货决策 (Order Qty)"
    )
    ax1.set_ylabel("数量 (件)", fontsize=11)
    ax1.set_title("图 1: 库存-补货-需求 决策控制轨迹", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    p1 = ax2.plot(
        steps, df["selling_price"], color="#ff7f0e", linewidth=2, label="Agent 决策售价 (Price)"
    )
    p2 = ax2_twin.plot(
        steps,
        df["sales_qty"],
        color="#9467bd",
        linestyle="-.",
        linewidth=1.8,
        label="实际成交销量 (Sales)",
    )
    ax2.set_ylabel("销售单价 (RM)", fontsize=11)
    ax2_twin.set_ylabel("实际销量 (件)", fontsize=11)
    ax2.set_title("图 2: 动态定价控制与市场响应曲线", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    lines = p1 + p2
    ax2.legend(lines, [line.get_label() for line in lines], loc="upper right", frameon=True)

    ax3 = axes[2]
    ax3.plot(
        steps,
        df["cum_net_profit"],
        color="#8c564b",
        linewidth=2.5,
        label="累计净收益 (Cumulative P&L)",
    )
    ax3.plot(steps, df["storage_cost"], color="#e377c2", linestyle=":", label="单步仓储成本")
    ax3.plot(steps, df["stockout_cost"], color="#7f7f7f", linestyle=":", label="单步缺货罚金")
    ax3.set_xlabel("时间步 (Day / Step)", fontsize=11)
    ax3.set_ylabel("金额 (RM)", fontsize=11)
    ax3.set_title("图 3: 累计财务损益与运营成本监控", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"评估曲线图已保存至: {save_path}")
    plt.close(fig)


def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    paths = get_paths(config, config_path)
    sac_cfg = get_section(config, "sac_agent")
    eval_cfg = get_section(
        config,
        "evaluation",
        {"difficulty": "easy", "use_final_model": False, "seed": 42},
    )

    eval_difficulty = eval_cfg.get("difficulty", "easy")
    eval_seed = int(eval_cfg.get("seed", 42))
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if eval_cfg.get("use_final_model", False):
        weight_path = model_path(paths, final=True)
    else:
        weight_path = model_path(paths, difficulty=eval_difficulty)

    env = ShopeeSupplyChainEnv(config_path=config_path, difficulty=eval_difficulty)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = load_trained_agent(weight_path, state_dim, action_dim, sac_cfg)

    print(f"开始在 [{eval_difficulty.upper()}] 难度上运行策略评估 | seed={eval_seed}...")
    df_history = run_evaluation_episode(env, agent, seed=eval_seed)

    print("\n" + "=" * 50)
    print("Episode 评估财务与运营统计摘要")
    print("=" * 50)
    print(f"总累计净收益 (Total P&L) : RM {df_history['cum_net_profit'].iloc[-1]:,.2f}")
    print(f"平均单期销售单价 (Avg Price): RM {df_history['selling_price'].mean():.2f}")
    print(f"平均单期期末库存 (Avg Stock): {df_history['inventory'].mean():.1f} 件")
    print(f"累计仓储总成本 (Storage Cost): RM {df_history['storage_cost'].sum():,.2f}")
    print(f"累计缺货总罚金 (Stockout Cost): RM {df_history['stockout_cost'].sum():,.2f}")
    print("=" * 50 + "\n")

    plot_decision_curves(df_history, save_path=paths["eval_decision_curves"])


if __name__ == "__main__":
    main()
