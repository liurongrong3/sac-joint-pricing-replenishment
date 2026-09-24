from collections import deque
import json
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config_loader import DEFAULT_CONFIG_PATH


class ShopeeSupplyChainEnv(gym.Env):
    """Shopee 马来站猫粮供应链决策环境 (动态定价 + 在途补货联合优化)"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path=DEFAULT_CONFIG_PATH,
        difficulty="easy",
    ):
        super(ShopeeSupplyChainEnv, self).__init__()

        self.difficulty = difficulty

        # ------------------------------------------
        # 1. 自动加载数据拟合参数与供应链基础设定
        # ------------------------------------------
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = json.load(f)
            config = full_config.get("environment", full_config)
            # 兼容 reward_shaping / reward_shaping 两种命名
            reward_shaping = full_config.get("reward_shaping") or full_config.get(
                "reward_shaping", {}
            )
            action_constraints = {
                k: v
                for k, v in full_config.get("action_constraints", {}).items()
                if not k.startswith("_")
            }
            self.difficulty_cfg = {
                k: v
                for k, v in full_config.get("difficulty", {}).items()
                if not k.startswith("_")
            }

            # 读取提前期 L (Lead Time) 和 最大仿真天数
            self.L = config.get("lead_time", 14)
            self.max_steps = config.get("max_steps", 90)

            # 读取价格销售曲线参数
            self.fit_A = config.get("fit_A", 50.75390712781621)
            self.fit_gamma = config.get("fit_gamma", 1.8138224003791896)
            self.base_comp_price = config.get("p_comp", 35.0)
            self.comp_price = self.base_comp_price
            self.initial_inventory = float(
                config.get("initial_inventory", config.get("inventory", 100.0))
            )
            self.inventory = self.initial_inventory

            # 读取动作映射价格和补货量参数
            self.min_price = config.get("min_price", 15.0)
            self.max_price = config.get("max_price", 65.0)
            self.max_order_qty = config.get("max_order_qty", 200.0)
            self.min_order_threshold = config.get("min_order_threshold", 10.0)

            # 读取 P&L 财务成本设定
            self.cogs = config.get("cogs", 12.0)
            self.headhaul_shipping = config.get("headhaul_shipping", 3.0)
            self.fixed_order_cost = config.get("fixed_order_cost", 50.0)
            self.storage_normal_capacity = float(
                config.get("storage_normal_capacity", 200.0)
            )
            self.storage_fee_normal = config.get("storage_fee_normal", 0.05)
            self.storage_fee_overstock = config.get("storage_fee_overstock", 0.25)
            self.stockout_penalty = config.get("stockout_penalty", 15.0)
            self.ending_inventory_salvage_ratio = float(
                config.get("ending_inventory_salvage_ratio", 1.0)
            )

            # RL 训练奖励塑形系数（兼容 w_* / w_* 命名）
            self.w_storage = reward_shaping.get(
                "w_storage", reward_shaping.get("w_storage", 1.2)
            )
            self.w_stockout = reward_shaping.get(
                "w_stockout", reward_shaping.get("w_stockout", 1.1)
            )
            self.w_price_extreme = reward_shaping.get(
                "w_price_extreme", reward_shaping.get("w_price_extreme", 80.0)
            )

            # 动作安全约束
            self.min_margin = float(
                action_constraints.get(
                    "min_margin", action_constraints.get("min_margin", 5.0)
                )
            )
            self.max_inventory_position = float(
                action_constraints.get(
                    "max_inventory_position",
                    action_constraints.get("max_inventory_position", 650.0),
                )
            )
            self.price_band_half_width = float(
                action_constraints.get(
                    "price_band_half_width",
                    action_constraints.get("price_band_half_width", 0.45),
                )
            )

            print(
                f"🔗 成功加载配置文件 {config_path} | 提前期 L={self.L}天 | fit_A={self.fit_A:.2f}, fit_gamma={self.fit_gamma:.2f}"
            )
        else:
            # 若找不到配置文件，使用兜底默认值
            self.L = 14
            self.max_steps = 90

            self.fit_A = 50.75390712781621
            self.fit_gamma = 1.8138224003791896
            self.base_comp_price = 35.0
            self.comp_price = 35.0
            self.initial_inventory = 100.0
            self.inventory = 100.0

            self.min_price = 15.0
            self.max_price = 65.0
            self.max_order_qty = 200.0
            self.min_order_threshold = 10.0

            # P&L 财务成本设定
            self.cogs = 12.0  # 采购单价 (RM 12/kg)
            self.headhaul_shipping = 3.0  # 头程运费单价 (RM 3/kg)
            self.fixed_order_cost = 50.0  # 每次起运固定杂费
            self.storage_normal_capacity = 200.0  # 正常仓储容量上限（件）
            self.storage_fee_normal = 0.05  # 正常仓储费 (RM 0.05/件/天)
            self.storage_fee_overstock = 0.25  # 积压仓储费（超出容量部分）
            self.stockout_penalty = 15.0  # 缺货机会成本惩罚 (RM 15/件)
            self.ending_inventory_salvage_ratio = 1.0
            self.w_storage = 1.2
            self.w_stockout = 1.1
            self.w_price_extreme = 80.0
            self.min_margin = 5.0
            self.max_inventory_position = 650.0
            self.price_band_half_width = 0.45
            self.difficulty_cfg = {}

            print("⚠️ 未找到 config.json，将使用默认拟合参数运行。")

        # 单位进货成本：用于定价成本地板
        self.unit_landed_cost = float(self.cogs + self.headhaul_shipping)

        # ------------------------------------------
        # 2. 定义 MDP 空间 (Spaces)
        # ------------------------------------------
        # 动作空间：2维连续动作 [-1.0, 1.0]
        # a[0]: 对应定价比例映射
        # a[1]: 对应补货量映射
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 状态空间维度：1(当前库存) + L(在途队列) + 1(竞品价格) + 1(大促标记)
        obs_dim = 1 + self.L + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 重置环境状态
        self.reset()

    def set_difficulty(self, difficulty: str):
        """用于课程学习 (Curriculum Learning) 动态切换难度桶"""
        assert difficulty in ("easy", "medium", "hard"), (
            f"未知难度 '{difficulty}'，仅支持 easy / medium / hard"
        )
        self.difficulty = difficulty

    def _apply_market_dynamics(self):
        """
        按商业场景设置当日市场需求波动与竞争环境。
        数值全部来自 config.json → difficulty.{easy,medium,hard}；
        竞品基准价来自 environment.p_comp。

        easy   —— 平稳日常月：需求噪声低，竞品价固定，无平台大促
        medium —— 常规竞争月：周末流量 + 竞品小幅调价，无 9.9/10.10
        hard   —— 大促冲刺月：平台大促窗、竞品跟价打折、噪声更大

        约定：promo_flag 只表示「平台级大促」；
        medium 的周末流量只抬需求倍率，不污染 promo_flag。
        """
        cfg = self.difficulty_cfg.get(self.difficulty, {})
        base_price = self.base_comp_price

        if self.difficulty == "easy":
            noise_std = cfg.get("noise_std", 0.05)
            self.promo_flag = 0.0
            self.comp_price = float(base_price)
            demand_multiplier = cfg.get("demand_multiplier", 1.0)

        elif self.difficulty == "medium":
            noise_std = cfg.get("noise_std", 0.10)
            self.promo_flag = 0.0
            jitter_std = cfg.get("comp_price_jitter_std", 1.5)
            price_min = cfg.get("comp_price_min", 30.0)
            price_max = cfg.get("comp_price_max", 40.0)
            self.comp_price = float(
                np.clip(
                    base_price + float(self.np_random.normal(0, jitter_std)),
                    price_min,
                    price_max,
                )
            )
            weekend_days = set(cfg.get("weekend_days", [5, 6]))
            is_weekend = (self.current_day % 7) in weekend_days
            demand_multiplier = (
                cfg.get("weekend_demand_multiplier", 1.2)
                if is_weekend
                else cfg.get("demand_multiplier", 1.0)
            )

        else:  # hard
            noise_std = cfg.get("noise_std", 0.20)
            cycle = cfg.get("promo_cycle_days", 30)
            window = set(cfg.get("promo_window_days", [28, 29, 0, 1]))
            self.promo_flag = (
                1.0 if (self.current_day % cycle in window) else 0.0
            )
            promo_discount = (
                cfg.get("promo_discount", 0.85)
                if self.promo_flag == 1.0
                else 1.0
            )
            price_noise = cfg.get("promo_price_noise_std", 1.0)
            # 大促日打折；非大促日仍保留小噪声，模拟价格战日常扰动
            self.comp_price = float(
                base_price * promo_discount
                + float(self.np_random.normal(0, price_noise))
            )
            demand_multiplier = (
                cfg.get("promo_demand_multiplier", 3.0)
                if self.promo_flag == 1.0
                else cfg.get("demand_multiplier", 1.0)
            )

        return noise_std, demand_multiplier

    def reset(self, seed=None, options=None):
        # seed 会种 self.np_random；step 里需求/竞品噪声都走它，便于多策略配对对照
        super().reset(seed=seed)

        self.current_day = 0
        self.inventory = float(self.initial_inventory)
        self.pipeline_queue = deque([0.0] * self.L, maxlen=self.L)
        self.promo_flag = 0.0

        # 期初存货资本：首步从 P&L 扣减，与期末残值配对，避免「白捡库存」
        self._opening_inventory_charge = (
            float(self.initial_inventory) * self.unit_landed_cost
        )
        self._opening_charge_pending = True

        return self._get_obs(), {}

    def _get_obs(self):
        """拼接状态向量"""
        pipeline_list = list(self.pipeline_queue)
        obs = (
            [self.inventory]
            + pipeline_list
            + [self.comp_price, self.promo_flag]
        )
        return np.array(obs, dtype=np.float32)

    def _inventory_position(self):
        """现货 + 在途 = 库存头寸（下单前视角）。"""
        return float(self.inventory + sum(self.pipeline_queue))

    def _map_actions(self, action):
        """
        将 SAC 的 [-1,1] 动作映射为业务量，并施加安全约束：
        1) 售价不低于 单位成本+min_margin，且靠近竞品价的带宽内（抑制 15/65 bang-bang）
        2) 订货受 max_inventory_position 限制（抑制 L 天延迟下连续超订）
        """
        a_p, a_q = float(action[0]), float(action[1])

        # ---- 价格：先线性映射，再夹到 [成本地板, 带宽∩业务上下限] ----
        raw_price = self.min_price + (a_p + 1.0) * 0.5 * (
            self.max_price - self.min_price
        )
        cost_floor = self.unit_landed_cost + self.min_margin
        band_lo = self.comp_price * (1.0 - self.price_band_half_width)
        band_hi = self.comp_price * (1.0 + self.price_band_half_width)
        price_lo = max(self.min_price, cost_floor, band_lo)
        price_hi = min(self.max_price, band_hi)
        if price_lo > price_hi:
            # 带宽与成本地板冲突时，优先保证不亏本卖
            price_lo, price_hi = cost_floor, max(cost_floor, self.max_price)
        price = float(np.clip(raw_price, price_lo, price_hi))

        # ---- 补货：映射后按剩余头寸容量截断 ----
        raw_qty = (a_q + 1.0) * 0.5 * self.max_order_qty
        order_qty = (
            float(np.round(raw_qty))
            if raw_qty >= self.min_order_threshold
            else 0.0
        )
        # 注意：step 里到货先于下单；此处用「到货前」头寸做保守上限，
        # 真正截断在 step 里用到货后的头寸再算一次。
        remaining = max(0.0, self.max_inventory_position - self._inventory_position())
        order_qty = float(min(order_qty, remaining))
        if order_qty < self.min_order_threshold:
            order_qty = 0.0

        return price, order_qty

    def step(self, action):
        self.current_day += 1

        # ------------------------------------------
        # Step A: 到货先入账，再用更新后的头寸约束当日订货
        # ------------------------------------------
        arrived_qty = self.pipeline_queue.popleft()
        self.inventory += arrived_qty

        price, order_qty = self._map_actions(action)
        # 到货后再次按剩余容量截断，避免「货刚到又猛订」
        remaining = max(
            0.0,
            self.max_inventory_position
            - float(self.inventory + sum(self.pipeline_queue)),
        )
        order_qty = float(min(order_qty, remaining))
        if order_qty < self.min_order_threshold:
            order_qty = 0.0

        self.pipeline_queue.append(order_qty)

        # ------------------------------------------
        # Step B: 需求计算与市场环境动态
        # ------------------------------------------
        noise_std, demand_multiplier = self._apply_market_dynamics()

        predict_demand = (
            self.fit_A
            * ((price / self.comp_price) ** (-self.fit_gamma))
            * demand_multiplier
        )

        noise = float(self.np_random.normal(0, noise_std))
        actual_demand = max(0, int(round(predict_demand * np.exp(noise))))

        # ------------------------------------------
        # Step C: 履约结算 (Sales & Inventory)
        # ------------------------------------------
        actual_sales = int(min(self.inventory, actual_demand))
        stockout_qty = int(actual_demand - actual_sales)
        self.inventory -= actual_sales

        # ------------------------------------------
        # Step D: 零售会计口径的单步 P&L / 财富变化
        #   - 下单日：只记固定起运杂费（变动进货成本不进费用，视为存货资产）
        #   - 售出日：结转 landed COGS = 销量 × (cogs + headhaul)
        #   - 期初：扣减期初存货账面价值（与期末残值配对，保证恒等式）
        #   - 期末：现货+在途按残值比例回收
        #   恒等式（ratio=1）：Σ(rev - cogs_sold - fixed - storage - stockout)
        #                      - opening + ending ≡ 现金流口径
        # ------------------------------------------

        # 销售收入：当日实际卖掉的件数 × 当日售价
        revenue = actual_sales * price

        # 固定起运/下单杂费：有订货才扣一次；与订货量无关（变动成本不在此扣）
        fixed_order_fee = self.fixed_order_cost if order_qty > 0 else 0.0
        # 已售商品成本(COGS)：卖掉才结转；unit_landed_cost = 采购成本 + 头程运费
        cogs_sold = actual_sales * self.unit_landed_cost
        # 兼容旧日志字段：当日「固定费 + 已售 COGS」合计（≠ 下单全额进货成本）
        order_cost = fixed_order_fee + cogs_sold

        # 仓储费：按当日售出后剩余现货计；≤容量用正常费率，超出部分用超储费率
        cap = self.storage_normal_capacity
        if self.inventory <= cap:
            storage_cost = self.inventory * self.storage_fee_normal
        else:
            storage_cost = (
                cap * self.storage_fee_normal
                + (self.inventory - cap) * self.storage_fee_overstock
            )

        # 缺货罚金：有需求没货的件数 × 单位机会成本/信誉损失
        stockout_cost = stockout_qty * self.stockout_penalty

        terminated = self.current_day >= self.max_steps
        truncated = False

        # 期初存货账面扣减：仅 Episode 第 1 步；避免「白捡开局库存」又拿期末残值
        opening_charge = 0.0
        if self._opening_charge_pending:
            opening_charge = self._opening_inventory_charge
            self._opening_charge_pending = False

        # 期末存货残值：仅最后一天；现货+在途按 landed cost × salvage_ratio 回收
        inventory_salvage = 0.0
        if terminated:
            ending_position = float(self.inventory + sum(self.pipeline_queue))
            inventory_salvage = (
                ending_position
                * self.unit_landed_cost
                * self.ending_inventory_salvage_ratio
            )

        # A. 真实业务 P&L（评估/日志用；权重均为 1，无额外 shaping）
        # 毛利：卖货毛利 − 当日固定下单费（未扣仓储/缺货）
        gross_profit = revenue - cogs_sold - fixed_order_fee
        # 净利：毛利 − 仓储 − 缺货 − 期初扣减 + 期末残值（报表口径）
        net_profit = (
            gross_profit
            - storage_cost
            - stockout_cost
            - opening_charge
            + inventory_salvage
        )

        # B. RL 训练奖励（可与 net_profit 不一致：仓储/缺货加权 + 极端定价惩罚）
        # 售价相对竞品偏离的二次惩罚：只进 reward，不进真实 P&L
        price_dev = abs(price - self.comp_price) / max(self.comp_price, 1e-6)
        price_extreme_penalty = self.w_price_extreme * (price_dev ** 2)
        # SAC 优化目标：≈ net_profit，但 storage/stockout 乘 w_*，再减定价惩罚
        reward = (
            revenue
            - cogs_sold
            - fixed_order_fee
            - (self.w_storage * storage_cost)
            - (self.w_stockout * stockout_cost)
            - price_extreme_penalty
            - opening_charge
            + inventory_salvage
        )

        info = {
            "day": self.current_day,
            "order_cost": order_cost,              # 固定费 + 已售 COGS（兼容旧字段）
            "fixed_order_fee": fixed_order_fee,    # 下单固定杂费
            "cogs_sold": cogs_sold,                # 当日已售结转成本
            "storage_cost": storage_cost,          # 仓储费（原始金额）
            "stockout_cost": stockout_cost,        # 缺货罚金（原始金额）
            "opening_charge": opening_charge,      # 期初存货扣减（仅首日）
            "inventory_salvage": inventory_salvage,  # 期末残值回收（仅末日）
            "price": price,
            "order_qty": order_qty,
            "actual_demand": actual_demand,
            "actual_sales": actual_sales,
            "stockout_qty": stockout_qty,
            "ending_inventory": self.inventory,    # 当日结束后现货件数
            "revenue": revenue,                    # 销售收入
            "gross_profit": gross_profit,          # 毛利（未扣仓储/缺货）
            "net_profit": net_profit,              # 真实净利 / 评估用
            "reward_profit": reward,               # 训练用 reward（含 shaping）
            "price_extreme_penalty": price_extreme_penalty,  # 仅 reward 的定价惩罚
            "inventory_position": float(self.inventory + sum(self.pipeline_queue)),  # 现货+在途
        }

        return self._get_obs(), reward, terminated, truncated, info


# # =====================================================================
# # 🔍 随机策略 (Random Agent) 测试主逻辑
# # =====================================================================
# if __name__ == "__main__":
#     print("=" * 80)
#     print("🧪 开始校验 Shopee 供应链环境 (Random Agent - 90天仿真)")
#     print("=" * 80)

#     # 1. 实例化环境 (难度设为 hard)
#     env = ShopeeSupplyChainEnv(difficulty="hard")
#     obs, info = env.reset(seed=42)

#     # 2. 校验观察向量 (Observation Space)
#     print("\n🔍 【检查 1：观察向量 (Observation Vector) 解析】")
#     print(f"• 向量总维度: {obs.shape[0]} 维 (期望 17 维)")
#     print(f"• [0] 现货库存: {obs[0]:.1f} 件")
#     print(f"• [1:15] 14天在途队列: {obs[1:15]}")
#     print(f"• [15] 竞品价格: RM {obs[15]:.2f}")
#     print(f"• [16] 大促标记: {obs[16]}")
#     assert obs.shape[0] == 17, "❌ 状态向量维度异常，非 17 维！"

#     # 3. 统计变量初始化
#     total_revenue = 0.0
#     total_order_cost = 0.0
#     total_storage_cost = 0.0
#     total_stockout_cost = 0.0
#     total_net_profit = 0.0
#     stockout_days = 0

#     print("\n🚀 【检查 2：90 天连续运行采样日志 (每 15 天打印一次 + 大促日重点监控)】")
#     print("-" * 105)
#     print(
#         f"{'天数':^6}|{'定价(RM)':^10}|{'下单量':^8}|{'需求/销量':^12}|{'结余库存':^10}|{'缺货量':^8}|{'营业额(RM)':^12}|{'单日利润(RM)':^12}|{'大促'}"
#     )
#     print("-" * 105)

#     for day in range(1, 91):
#         # 随机采样动作
#         random_action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(random_action)

#         # 累计财务指标
#         total_revenue += info["revenue"]
#         total_order_cost += info["order_cost"]
#         total_storage_cost += info["storage_cost"]
#         total_stockout_cost += info["stockout_cost"]
#         total_net_profit += reward

#         if info["stockout_qty"] > 0:
#             stockout_days += 1

#         # 条件打印日志（每 15 天打印，或在第 28-31 天大促爆发期重点打印）
#         is_promo = obs[16] == 1.0
#         if day % 15 == 0 or day in [28, 29, 30, 31, 58, 59, 60, 61]:
#             promo_str = "🔥大促" if is_promo else "日常"
#             print(
#                 f"Day {day:02d} | RM {info['price']:6.2f} | {info['order_qty']:6.0f}件 | "
#                 f"{info['actual_demand']:4d} / {info['actual_sales']:4d} | {info['ending_inventory']:8.0f}件 | "
#                 f"{info['stockout_qty']:6d}件 | RM {info['revenue']:9.2f} | RM {reward:9.2f} | {promo_str}"
#             )

#         if terminated:
#             break

#     # 4. P&L 财务汇总报表
#     print("-" * 105)
#     print("\n📊 【检查 3：90 天最终 P&L 财务结算汇总报表】")
#     print("=" * 50)
#     print(f"💰 总营业收入 (Revenue)       : RM {total_revenue:>12,.2f}")
#     print(f"📦 总采购与运费成本 (Order Cost): RM {total_order_cost:>12,.2f}")
#     print(f"🏭 总仓储管理成本 (Storage Fee): RM {total_storage_cost:>12,.2f}")
#     print(f"⚠️ 总缺货损失惩罚 (Stockout)  : RM {total_stockout_cost:>12,.2f}")
#     print("=" * 50)
#     print(f"🏆 90天净利润 (Net Reward/P&L) : RM {total_net_profit:>12,.2f}")
#     print(f"🚨 缺货发生天数              : {stockout_days} / 90 天")
#     print("=" * 50)

#     # 校验数学等式恒等：Net Profit = Revenue - Order Cost - Storage Cost - Stockout Cost
#     calculated_pnl = (
#         total_revenue
#         - total_order_cost
#         - total_storage_cost
#         - total_stockout_cost
#     )
#     assert (
#         abs(calculated_pnl - total_net_profit) < 1e-2
#     ), "❌ P&L 财务平衡表等式校验失败！"
#     print("✅ P&L 财务平账校验成功！`Revenue - Costs - Penalties = Net Profit` 完全吻合。\n")


