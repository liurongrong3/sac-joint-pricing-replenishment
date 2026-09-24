"""
同一环境下对比 Random / ROP+跟价 / 滚动MILP / SAC，输出干净对比表。

全部参数在 config.json → baseline_comparison（及 evaluation / paths）。
用法：python compare_baselines.py
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from baselines import RandomBaseline, ROPMatchPriceBaseline, SACBaseline
from rolling_milp import RollingHorizonMILPBaseline
from config_loader import (
    DEFAULT_CONFIG_PATH,
    get_paths,
    get_section,
    load_config,
    model_path,
)
from sac_agent import SACAgent
from shopee_env import ShopeeSupplyChainEnv

# SACAgent.__init__ 只吃这些键，避免 config 多字段报错
_SAC_INIT_KEYS = {
    "gamma",
    "tau",
    "lr",
    "hidden_dim",
    "reward_scale",
    "log_alpha_min",
    "log_alpha_max",
    "grad_clip_max_norm",
    "log_std_min",
    "log_std_max",
}


def _load_sac_agent(env, config_path: str, difficulty: str, use_final: bool):
    """加载评估用 SAC Actor；找不到权重返回 (None, path)。"""
    config = load_config(config_path)
    paths = get_paths(config, config_path)
    sac_cfg = {
        k: v
        for k, v in get_section(config, "sac_agent").items()
        if k in _SAC_INIT_KEYS
    }
    weight = model_path(paths, difficulty=difficulty, final=use_final)
    if not os.path.exists(weight):
        # 难度权重缺失时，尝试 final，再尝试旧版 sac_actor_best.pth
        fallbacks = [
            model_path(paths, final=True),
            os.path.join(paths["models_dir"], "sac_actor_best.pth"),
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                weight = fb
                break
        else:
            return None, weight

    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **sac_cfg,
    )
    state_dict = torch.load(weight, map_location="cpu")
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    return agent, weight


def run_episode(
    env: ShopeeSupplyChainEnv,
    action_fn: Callable,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """跑完整 Episode，汇总真实业务指标（用 info['net_profit']，非 shaping reward）。"""
    obs, _ = env.reset(seed=seed)
    cum_pnl = 0.0
    storage = 0.0
    stockout_cost = 0.0
    stockout_days = 0
    inventories: List[float] = []
    prices: List[float] = []
    max_drop_vs_comp = 0.0

    for _ in range(env.max_steps):
        action = action_fn(env, obs)
        obs, _reward, terminated, truncated, info = env.step(action)

        cum_pnl += float(info.get("net_profit", 0.0))
        storage += float(info.get("storage_cost", 0.0))
        stockout_cost += float(info.get("stockout_cost", 0.0))
        if float(info.get("stockout_qty", 0.0)) > 0:
            stockout_days += 1

        inv = float(info.get("ending_inventory", 0.0))
        price = float(info.get("price", 0.0))
        inventories.append(inv)
        prices.append(price)

        # 相对当日竞品的最大降幅（正数=卖得比竞品低）
        comp = float(env.comp_price)
        if comp > 1e-6:
            drop = max(0.0, (comp - price) / comp)
            max_drop_vs_comp = max(max_drop_vs_comp, drop)

        if terminated or truncated:
            break

    return {
        "net_pnl": cum_pnl,
        "stockout_days": float(stockout_days),
        "avg_inventory": float(np.mean(inventories)) if inventories else 0.0,
        "storage_cost": storage,
        "stockout_cost": stockout_cost,
        "avg_price": float(np.mean(prices)) if prices else 0.0,
        "max_drop_vs_comp": max_drop_vs_comp,
        "steps": float(len(inventories)),
    }


def aggregate_runs(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "net_pnl",
        "stockout_days",
        "avg_inventory",
        "storage_cost",
        "stockout_cost",
        "avg_price",
        "max_drop_vs_comp",
    ]
    out = {"n_episodes": float(len(rows))}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    return out


def evaluate_policy(
    name: str,
    env: ShopeeSupplyChainEnv,
    action_fn: Callable,
    seeds: List[int],
) -> Dict[str, float]:
    rows = [run_episode(env, action_fn, seed=s) for s in seeds]
    summary = aggregate_runs(rows)
    summary["strategy"] = name
    return summary


def format_comparison_table(df: pd.DataFrame) -> str:
    """打印用的紧凑表。"""
    cols = [
        ("strategy", "策略", None),
        ("net_pnl_mean", "净P&L均值(RM)", "{:,.1f}"),
        ("net_pnl_std", "P&L±std", "{:,.1f}"),
        ("stockout_days_mean", "缺货天数", "{:.1f}"),
        ("avg_inventory_mean", "平均库存", "{:.1f}"),
        ("storage_cost_mean", "仓储费", "{:,.1f}"),
        ("stockout_cost_mean", "缺货罚金", "{:,.1f}"),
        ("avg_price_mean", "均价(RM)", "{:.2f}"),
        ("max_drop_vs_comp_mean", "最大跟跌%", None),
    ]
    headers = [c[1] for c in cols]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for _, row in df.iterrows():
        cells = []
        for key, _label, fmt in cols:
            val = row[key]
            if key == "max_drop_vs_comp_mean":
                # 不用 "{:.1%}"：行尾的 % 在 Windows 终端会吞掉后面的换行，
                # 把下一行策略名粘在 2.0% 后面。
                cells.append("{:.1f}%".format(float(val) * 100.0))
            elif fmt is None:
                cells.append(str(val))
            else:
                cells.append(fmt.format(val))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    eval_cfg = get_section(
        config, "evaluation", {"difficulty": "easy", "use_final_model": False}
    )
    cmp_cfg = get_section(
        config,
        "baseline_comparison",
        {
            "difficulty": None,
            "episodes": 5,
            "seed": 42,
            "skip_sac": True,
            "skip_milp": False,
            "match_ratio": 0.98,
            "safety_days": 3.0,
            "review_horizon": 1.0,
            "horizon": 21,
            "milp_time_limit": 5.0,
            "milp_gap_rel": 0.01,
        },
    )
    # 字典类型的get函数
    difficulty = cmp_cfg.get("difficulty") or eval_cfg.get("difficulty", "easy")
    use_final = bool(eval_cfg.get("use_final_model", False))
    episodes = int(cmp_cfg.get("episodes", 5))
    seed0 = int(cmp_cfg.get("seed", 42))
    skip_sac = bool(cmp_cfg.get("skip_sac", True))
    skip_milp = bool(cmp_cfg.get("skip_milp", False))
    match_ratio = float(cmp_cfg.get("match_ratio", 0.98))
    safety_days = float(cmp_cfg.get("safety_days", 3.0))
    review_horizon = float(cmp_cfg.get("review_horizon", 1.0))
    milp_horizon = int(cmp_cfg.get("horizon", 21))
    milp_time_limit = float(cmp_cfg.get("milp_time_limit", 5.0))
    milp_gap_rel = float(cmp_cfg.get("milp_gap_rel", 0.01))

    paths = get_paths(config, config_path, mkdir=True)
    seeds = [seed0 + i for i in range(episodes)]
    env = ShopeeSupplyChainEnv(config_path=config_path, difficulty=difficulty)

    print("=" * 72)
    print(
        f"策略对比评测 | difficulty={difficulty} | episodes={episodes} "
        f"| seeds={seeds[0]}..{seeds[-1]} | skip_sac={skip_sac} | skip_milp={skip_milp}"
    )
    print("=" * 72)

    results: List[Dict[str, float]] = []

    random_policy = RandomBaseline()
    print("\n▶ 运行 Random ...")
    results.append(
        evaluate_policy(
            random_policy.name,
            env,
            lambda e, o: random_policy.select_action(e, o),
            seeds,
        )
    )

    rop_policy = ROPMatchPriceBaseline(
        match_ratio=match_ratio,
        safety_days=safety_days,
        review_horizon=review_horizon,
    )
    print("▶ 运行 ROP+MatchPrice ...")
    results.append(
        evaluate_policy(
            rop_policy.name,
            env,
            lambda e, o: rop_policy.select_action(e, o),
            seeds,
        )
    )

    if not skip_milp:
        milp_policy = RollingHorizonMILPBaseline(
            match_ratio=match_ratio,
            horizon=milp_horizon,
            time_limit=milp_time_limit,
            gap_rel=milp_gap_rel,
        )
        print(
            f"▶ 运行 MILP+MatchPrice | horizon={milp_horizon} "
            f"| timeLimit={milp_time_limit}s | gapRel={milp_gap_rel}"
        )
        results.append(
            evaluate_policy(
                milp_policy.name,
                env,
                lambda e, o: milp_policy.select_action(e, o),
                seeds,
            )
        )

    if not skip_sac:
        agent, weight = _load_sac_agent(env, config_path, difficulty, use_final)
        if agent is None:
            print(f"⚠ 未找到 SAC 权重: {weight}")
            print(
                "  已跳过 SAC。请先 python train.py，"
                "或在 config.json 设 baseline_comparison.skip_sac=true。"
            )
        else:
            sac_policy = SACBaseline(agent)
            print(f"▶ 运行 SAC | 权重: {weight} | episodes={episodes}")
            results.append(
                evaluate_policy(
                    sac_policy.name,
                    env,
                    lambda e, o: sac_policy.select_action(e, o),
                    seeds,
                )
            )

    df = pd.DataFrame(results)
    front = ["strategy", "n_episodes"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    out_csv = paths["baseline_comparison_csv"]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("干净对比表（均值；多 seed 时附带 std）")
    print("=" * 72)
    # 逐行打印，避免整段字符串里的 % 被终端当格式符吃掉换行
    for line in format_comparison_table(df).splitlines():
        print(line)
    print("=" * 72)
    print(f"\n已保存 CSV: {out_csv}")
    print(
        "说明: 指标均为真实 P&L 口径（net_profit），不含 reward shaping；"
        "参数请改 config.json → baseline_comparison。"
    )


if __name__ == "__main__":
    main()
