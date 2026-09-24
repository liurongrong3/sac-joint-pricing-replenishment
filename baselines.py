"""
规则 / 随机 / 滚动 MILP 基线：输出与环境一致的 [-1, 1] 连续动作。

- RandomBaseline: action_space.sample()，作下界对照
- ROPMatchPriceBaseline: 再订货点补货 + 竞品跟价（产业常识 L1）
- RollingHorizonMILPBaseline: 规则跟价 + 滚动时域 MILP 补货（见 rolling_milp.py）
"""

from __future__ import annotations

import numpy as np


def encode_business_action(env, price: float, order_qty: float) -> np.ndarray:
    """把业务量 (售价, 订货量) 反解为环境所需的 [-1, 1] 动作。"""
    price_span = max(float(env.max_price - env.min_price), 1e-6)
    a_p = 2.0 * (float(price) - float(env.min_price)) / price_span - 1.0
    a_q = 2.0 * float(order_qty) / max(float(env.max_order_qty), 1e-6) - 1.0
    return np.array(
        [np.clip(a_p, -1.0, 1.0), np.clip(a_q, -1.0, 1.0)],
        dtype=np.float32,
    )


def inventory_position_from_obs(obs: np.ndarray, lead_time: int) -> float:
    """现货 + 在途头寸（与 env._inventory_position 同构）。"""
    on_hand = float(obs[0])
    in_transit = float(np.sum(obs[1 : 1 + lead_time]))
    return on_hand + in_transit


class RandomBaseline:
    """随机策略：环境校验 / 对比表下界。"""

    name = "Random"

    def reset(self):
        return None

    def select_action(self, env, obs: np.ndarray) -> np.ndarray:
        return env.action_space.sample()


class ROPMatchPriceBaseline:
    """
    规则基线：竞品跟价 + (s, S) 再订货点补货。

    定价:
        target = p_comp * match_ratio，再夹到成本地板与业务价带
    补货:
        μ ≈ fit_A * (p / p_comp)^(-γ) * 需求倍率
        SS = safety_days * μ
        ROP = μ * L + SS
        S   = μ * (L + review_horizon) + SS
        若 IP < ROP: 订 min(max_order, S - IP)
    """

    name = "ROP+MatchPrice"

    def __init__(
        self,
        match_ratio: float = 0.98,
        safety_days: float = 3.0,
        review_horizon: float = 1.0,
    ):
        self.match_ratio = float(match_ratio)
        self.safety_days = float(safety_days)
        self.review_horizon = float(review_horizon)

    def reset(self):
        return None

    def _expected_daily_demand(self, env, price: float) -> float:
        ratio = max(float(price) / max(float(env.comp_price), 1e-6), 1e-6)
        mu = float(env.fit_A) * (ratio ** (-float(env.fit_gamma)))
        cfg = env.difficulty_cfg.get(env.difficulty, {})
        mu *= float(cfg.get("demand_multiplier", 1.0))
        # medium 周末：规则基线不知道「今天是否周末」的精确 weekday 时，
        # 用均值近似；hard 大促用 promo_flag 抬需求。
        if env.difficulty == "medium":
            weekend_mult = float(cfg.get("weekend_demand_multiplier", 1.2))
            weekend_days = cfg.get("weekend_days", [5, 6])
            weekend_frac = len(weekend_days) / 7.0
            mu *= (1.0 - weekend_frac) + weekend_frac * weekend_mult
        if float(env.promo_flag) > 0.5:
            mu *= float(cfg.get("promo_demand_multiplier", 3.0))
        return max(mu, 0.0)

    def _target_price(self, env) -> float:
        cost_floor = float(env.unit_landed_cost + env.min_margin)
        raw = float(env.comp_price) * self.match_ratio
        band_lo = float(env.comp_price) * (1.0 - float(env.price_band_half_width))
        band_hi = float(env.comp_price) * (1.0 + float(env.price_band_half_width))
        price_lo = max(float(env.min_price), cost_floor, band_lo)
        price_hi = min(float(env.max_price), band_hi)
        if price_lo > price_hi:
            price_lo, price_hi = cost_floor, max(cost_floor, float(env.max_price))
        return float(np.clip(raw, price_lo, price_hi))

    def _order_qty(self, env, obs: np.ndarray, price: float) -> float:
        mu = self._expected_daily_demand(env, price)
        L = float(env.L)
        ss = self.safety_days * mu #安全库存
        rop = mu * L + ss #再订货点
        target_s = mu * (L + self.review_horizon) + ss #含安全库存的订货上限
        ip = inventory_position_from_obs(obs, env.L) #现货+在途

        if ip >= rop:
            return 0.0

        qty = max(0.0, target_s - ip)
        # 与环境一致：头寸帽 + 单次上限 + 最小起订
        remaining = max(0.0, float(env.max_inventory_position) - ip)
        qty = min(qty, remaining, float(env.max_order_qty))
        if qty < float(env.min_order_threshold):
            return 0.0
        return float(np.round(qty))

    def select_action(self, env, obs: np.ndarray) -> np.ndarray:
        price = self._target_price(env)
        order_qty = self._order_qty(env, obs, price)
        return encode_business_action(env, price, order_qty)


class SACBaseline:
    """
    已训练 SAC 策略封装，接口与 Random / ROP 一致。
    评估时用确定性动作（evaluate=True）。
    """

    name = "SAC"

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        return None

    def select_action(self, env, obs: np.ndarray) -> np.ndarray:
        return self.agent.select_action(obs, evaluate=True)


def build_baseline(name: str, **kwargs):
    """按名称构造基线。"""
    key = name.strip().lower()
    if key in ("random", "rand"):
        return RandomBaseline()
    if key in ("rop", "rop+matchprice", "rop_match", "rule"):
        return ROPMatchPriceBaseline(
            **{
                k: kwargs[k]
                for k in ("match_ratio", "safety_days", "review_horizon")
                if k in kwargs
            }
        )
    if key in ("milp", "milp+matchprice", "rolling_milp", "rolling-horizon"):
        from rolling_milp import RollingHorizonMILPBaseline

        allowed = {"match_ratio", "horizon", "time_limit", "gap_rel"}
        return RollingHorizonMILPBaseline(
            **{k: kwargs[k] for k in allowed if k in kwargs}
        )
    if key == "sac":
        agent = kwargs.get("agent")
        if agent is None:
            raise ValueError("构建 SAC 基线需要传入 agent=...")
        return SACBaseline(agent)
    raise ValueError(f"未知基线 '{name}'，可选: random / rop / milp / sac")
