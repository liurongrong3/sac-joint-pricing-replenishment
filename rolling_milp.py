"""
滚动时域 MILP 补货（规则跟价 + 求解器订货）。

接口与 ROPMatchPriceBaseline 相同：select_action(env, obs) -> [-1,1] 动作。
每天根据当前现货/在途建未来 N 天模型，只执行当天订货量 q_t。
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from baselines import ROPMatchPriceBaseline, encode_business_action

try:
    import pulp

    _PULP_AVAILABLE = True
except ImportError:
    pulp = None  # type: ignore
    _PULP_AVAILABLE = False


class RollingHorizonMILPBaseline:
    """
    定价：与 ROP 同一套竞品跟价（价格不进求解器）。
    补货：滚动 MILP，窗口 N 天，只下发当天 q。
    """

    name = "MILP+MatchPrice"

    def __init__(
        self,
        match_ratio: float = 0.98,
        horizon: int = 21,
        time_limit: float = 5.0,
        gap_rel: float = 0.01,
    ):
        self.rop = ROPMatchPriceBaseline(match_ratio=match_ratio)
        self.horizon = int(horizon)
        self.time_limit = float(time_limit)
        self.gap_rel = float(gap_rel)
        self._warned_no_pulp = False
        self._warned_infeasible = False
        self.last_status: Optional[str] = None
        self.last_gap: Optional[float] = None

    def reset(self):
        self.last_status = None
        self.last_gap = None
        return None

    def _sim_day_multiplier(self, env, sim_day: int) -> float:
        """即将发生的环境日（step 内 current_day 自增后）的期望需求倍率。"""
        cfg = env.difficulty_cfg.get(env.difficulty, {})
        m = float(cfg.get("demand_multiplier", 1.0))
        if env.difficulty == "medium":
            weekend_days = set(cfg.get("weekend_days", [5, 6]))
            if (int(sim_day) % 7) in weekend_days:
                m *= float(cfg.get("weekend_demand_multiplier", 1.2))
        elif env.difficulty == "hard":
            cycle = int(cfg.get("promo_cycle_days", 30))
            window = set(cfg.get("promo_window_days", [28, 29, 0, 1]))
            if (int(sim_day) % cycle) in window:
                m *= float(cfg.get("promo_demand_multiplier", 3.0))
        return m

    def _expected_demand(self, env, price: float, sim_day: int) -> float:
        ratio = max(float(price) / max(float(env.comp_price), 1e-6), 1e-6)
        mu = float(env.fit_A) * (ratio ** (-float(env.fit_gamma)))
        return max(mu * self._sim_day_multiplier(env, sim_day), 0.0)

    def _clip_qty(self, env, obs: np.ndarray, qty: float) -> float:
        """与环境一致：头寸帽 + 单次上限 + 最小起订。"""
        L = int(env.L)
        on_hand = float(obs[0])
        pipeline = [float(x) for x in obs[1 : 1 + L]]
        ip = on_hand + float(sum(pipeline))
        remaining = max(0.0, float(env.max_inventory_position) - ip)
        qty = min(max(0.0, qty), remaining, float(env.max_order_qty))
        if qty < float(env.min_order_threshold):
            return 0.0
        return float(np.round(qty))

    def _solve_order_qty(self, env, obs: np.ndarray, price: float) -> Optional[float]:
        if not _PULP_AVAILABLE:
            if not self._warned_no_pulp:
                warnings.warn(
                    "未安装 PuLP，MILP 基线回退为 ROP。请 pip install pulp",
                    RuntimeWarning,
                )
                self._warned_no_pulp = True
            return None

        L = int(env.L)
        remaining_steps = max(1, int(env.max_steps) - int(env.current_day))
        N = min(self.horizon, remaining_steps)
        on_hand = float(obs[0])

        # 在途14天队列，长度为14，obs向量的第1个元素是现货库存，MILP需要这些来做决策
        pipeline: List[float] = [float(x) for x in obs[1 : 1 + L]]
        if len(pipeline) < L:
            pipeline = pipeline + [0.0] * (L - len(pipeline))

        # 即将 step 时 current_day 会 +1，与 _apply_market_dynamics 对齐
        day0 = int(env.current_day) + 1
        demands = [
            self._expected_demand(env, price, day0 + tau) for tau in range(N)
        ]

        c = float(env.unit_landed_cost)
        F = float(env.fixed_order_cost)
        b = float(env.stockout_penalty)
        h1 = float(env.storage_fee_normal)
        h2 = float(env.storage_fee_overstock)
        cap_store = float(env.storage_normal_capacity)
        h_over = h2 - h1
        Qmax = float(env.max_order_qty)
        qmin = float(env.min_order_threshold)        
        ip_max = float(env.max_inventory_position)
        overflow_pen = 1.0e6

        def arrival(tau: int):
            if tau < L:
                return pipeline[tau]
            return q[tau - L]

        def hist_in_transit_after_pop(tau: int) -> float:
            return float(sum(pipeline[k] for k in range(tau + 1, L)))

        prob = pulp.LpProblem("rolling_replenish", pulp.LpMaximize)#最大化问题，prob是整道规划题的变量、约束、目标
        q = [
            pulp.LpVariable(f"q_{tau}", lowBound=0, upBound=Qmax, cat="Integer")
            for tau in range(N)
        ]
        y = [pulp.LpVariable(f"y_{tau}", cat="Binary") for tau in range(N)]
        s = [pulp.LpVariable(f"s_{tau}", lowBound=0) for tau in range(N)]
        I = [pulp.LpVariable(f"I_{tau}", lowBound=0) for tau in range(N)]
        o = [pulp.LpVariable(f"o_{tau}", lowBound=0) for tau in range(N)]
        w = [pulp.LpVariable(f"w_{tau}", lowBound=0) for tau in range(N)]
        overflow = [pulp.LpVariable(f"ov_{tau}", lowBound=0) for tau in range(N)]

        obj = []
        for tau in range(N):
            on_hand_before = on_hand if tau == 0 else I[tau - 1]
            arr = arrival(tau)
            # 库存：期末 = 到货后 - 销量
            prob += I[tau] == on_hand_before + arr - s[tau]
            prob += s[tau] <= on_hand_before + arr
            prob += s[tau] <= demands[tau]
            prob += o[tau] == demands[tau] - s[tau]
            # MOQ：不下单则为 0，下单则 [qmin, Qmax]
            prob += q[tau] <= Qmax * y[tau]
            prob += q[tau] >= qmin * y[tau]
            # 阶梯仓储
            prob += w[tau] >= I[tau] - cap_store
            # 头寸帽（允许 overflow 以免历史头寸超限导致无解）
            committed = hist_in_transit_after_pop(tau)
            j0 = max(0, tau - L + 1)
            future_q = pulp.lpSum(q[j] for j in range(j0, tau)) if tau > 0 else 0
            ip_after = on_hand_before + arr + committed + future_q
            prob += ip_after + q[tau] <= ip_max + overflow[tau]

            margin = float(price) - c
            obj.append(
                margin * s[tau]
                - F * y[tau]
                - (h1 * I[tau] + h_over * w[tau])
                - b * o[tau]
                - overflow_pen * overflow[tau]
            )

        # 窗口期末残值：现货 + 仍在途（未到货的历史管道 + 未到货订单）
        salvage_pipe = hist_in_transit_after_pop(N - 1)
        salvage_q = pulp.lpSum(q[j] for j in range(max(0, N - L), N))
        obj.append(c * (I[N - 1] + salvage_pipe + salvage_q))
        prob += pulp.lpSum(obj)

        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=self.time_limit,
            gapRel=self.gap_rel,
        )
        status_code = prob.solve(solver)
        status_name = pulp.LpStatus.get(status_code, str(status_code))
        self.last_status = status_name
        self.last_gap = None

        q0 = pulp.value(q[0])
        if q0 is None:
            if not self._warned_infeasible:
                warnings.warn(
                    f"滚动 MILP 无可行解 (status={status_name})，当日回退 ROP。",
                    RuntimeWarning,
                )
                self._warned_infeasible = True
            return None
        return float(q0)

    def select_action(self, env, obs: np.ndarray) -> np.ndarray:
        price = self.rop._target_price(env)
        qty = self._solve_order_qty(env, obs, price)
        if qty is None:
            qty = self.rop._order_qty(env, obs, price)
        qty = self._clip_qty(env, obs, float(qty))
        return encode_business_action(env, price, qty)


def build_milp_baseline(**kwargs) -> RollingHorizonMILPBaseline:
    allowed = {"match_ratio", "horizon", "time_limit", "gap_rel"}
    return RollingHorizonMILPBaseline(
        **{k: v for k, v in kwargs.items() if k in allowed}
    )
