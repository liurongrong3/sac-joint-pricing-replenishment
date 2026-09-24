import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config_loader import (
    DEFAULT_CONFIG_PATH,
    get_paths,
    get_section,
    load_config,
    save_config,
    update_section,
)

# environment 分区：Pipeline 生成/拟合后写回，环境读取
ENV_KEYS = (
    "lead_time",
    "max_steps",
    "p_comp",
    "initial_inventory",
    "min_price",
    "max_price",
    "max_order_qty",
    "min_order_threshold",
    "cogs",
    "headhaul_shipping",
    "fixed_order_cost",
    "storage_normal_capacity",
    "storage_fee_normal",
    "storage_fee_overstock",
    "stockout_penalty",
    "ending_inventory_salvage_ratio",
)

# simulation 分区：仅数据生成 Pipeline 使用，不参与 RL 训练
SIM_KEYS = (
    "seed",
    "n_samples",
    "true_A",
    "true_gamma",
    "demand_noise_std",
    "demand_clip_min",
    "demand_clip_max",
)

DEFAULT_ENV = {
    "lead_time": 14,
    "max_steps": 90,
    "p_comp": 35.0,
    "initial_inventory": 100.0,
    "min_price": 15.0,
    "max_price": 65.0,
    "max_order_qty": 200.0,
    "min_order_threshold": 10.0,
    "cogs": 12.0,
    "headhaul_shipping": 3.0,
    "fixed_order_cost": 50.0,
    "storage_normal_capacity": 200.0,
    "storage_fee_normal": 0.05,
    "storage_fee_overstock": 0.25,
    "stockout_penalty": 15.0,
    "ending_inventory_salvage_ratio": 1.0,
}

DEFAULT_SIM = {
    "seed": 2026,
    "n_samples": 500,
    "true_A": 50.0,
    "true_gamma": 1.8,
    "demand_noise_std": 0.15,
    "demand_clip_min": 1,
    "demand_clip_max": 300,
}


class ShopeeDemandPipeline:
    """需求弹性拟合 Pipeline：从 config.json 读参，拟合结果写回 environment 分区。"""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, **overrides):
        self.config_path = config_path
        self._full_config = (
            load_config(config_path) if os.path.exists(config_path) else {}
        )
        self._paths = get_paths(self._full_config, config_path)

        env = {**DEFAULT_ENV, **get_section(self._full_config, "environment")}
        sim = {**DEFAULT_SIM, **get_section(self._full_config, "simulation")}

        for key, value in overrides.items():
            if key in ENV_KEYS:
                env[key] = value
            elif key in SIM_KEYS:
                sim[key] = value

        self._env = env
        self._sim = sim
        for key in ENV_KEYS:
            setattr(self, key, env[key])
        for key in SIM_KEYS:
            setattr(self, key, sim[key])

        self.df: pd.DataFrame | None = None
        self.fit_A: float | None = None
        self.fit_gamma: float | None = None

    @staticmethod
    def demand_function(
        p: np.ndarray | float, A: float, gamma: float, p_comp: float
    ) -> np.ndarray | float:
        """核心需求公式: Demand = A * (Price / P_comp) ^ (-gamma)"""
        return A * ((p / p_comp) ** (-gamma))

    def generate_data(self, save_csv: str | None = None) -> pd.DataFrame:
        """生成模拟市场交互数据。"""
        save_csv = save_csv if save_csv is not None else self._paths["simulated_csv"]
        np.random.seed(self.seed)

        prices = np.random.uniform(self.min_price, self.max_price, self.n_samples)
        noise = np.random.normal(0, self.demand_noise_std, self.n_samples)
        demand = (
            self.true_A
            * ((prices / self.p_comp) ** (-self.true_gamma))
            * np.exp(noise)
        )
        demand = np.clip(
            demand, self.demand_clip_min, self.demand_clip_max
        ).astype(int)

        self.df = pd.DataFrame(
            {
                "my_price_rm": np.round(prices, 2),
                "comp_price_rm": self.p_comp,
                "daily_demand": demand,
            }
        )

        if save_csv:
            self.df.to_csv(save_csv, index=False)
            print(
                f"成功生成 {self.n_samples} 条模拟市场交互数据，已保存至: {save_csv}"
            )

        return self.df

    def fit_elasticity(self) -> Tuple[float, float]:
        """基于 Scipy 非线性最小二乘法拟合需求弹性曲线。"""
        if self.df is None:
            raise ValueError("数据未初始化，请先运行 generate_data()！")

        def fit_target_func(p, A, gamma):
            return self.demand_function(p, A, gamma, self.p_comp)

        popt, _ = curve_fit(
            fit_target_func, self.df["my_price_rm"], self.df["daily_demand"]
        )
        self.fit_A, self.fit_gamma = float(popt[0]), float(popt[1])

        print("\n需求弹性拟合结果：")
        print(f"   - 基础日需求量 (A): {self.fit_A:.2f} 件")
        print(f"   - 价格弹性系数 (Gamma): {self.fit_gamma:.2f}")
        print(
            f"   - 拟合公式: Demand = {self.fit_A:.2f} * (Price / {self.p_comp})^(-{self.fit_gamma:.2f})"
        )

        return self.fit_A, self.fit_gamma

    def export_config(
        self, config_path: str | None = None
    ) -> Dict[str, Any]:
        """将拟合结果与环境参数写回 config.json 的 environment 分区。"""
        if self.fit_A is None or self.fit_gamma is None:
            raise ValueError("未找到拟合参数，请先执行 fit_elasticity()！")

        config_path = config_path or self.config_path
        env_updates = {key: getattr(self, key) for key in ENV_KEYS}
        env_updates.update(
            {
                "fit_A": self.fit_A,
                "fit_gamma": self.fit_gamma,
                "inventory": self.initial_inventory,
            }
        )

        update_section(self._full_config, "environment", env_updates)
        update_section(
            self._full_config,
            "simulation",
            {key: getattr(self, key) for key in SIM_KEYS},
        )
        save_config(self._full_config, config_path)

        print(f"配置已写回: {config_path} (environment.fit_A / fit_gamma 已更新)")
        return get_section(self._full_config, "environment")

    def plot_curve(self, save_path: str | None = None) -> None:
        """绘制并保存需求弹性拟合曲线。"""
        save_path = save_path if save_path is not None else self._paths["demand_curve_plot"]
        if self.df is None or self.fit_A is None:
            raise ValueError("未找到有效数据或拟合参数，无法绘图！")

        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.df["my_price_rm"],
            self.df["daily_demand"],
            alpha=0.4,
            color="orange",
            label="Simulated Market Data",
        )

        p_range = np.linspace(self.min_price, self.max_price, 100)
        predicted_demand = self.demand_function(
            p_range, self.fit_A, self.fit_gamma, self.p_comp
        )

        plt.plot(
            p_range,
            predicted_demand,
            "r-",
            lw=2,
            label=rf"Fitted Elasticity Curve ($\gamma={self.fit_gamma:.2f}$)",
        )
        plt.title("Shopee MY Cat Food: Price vs. Daily Demand Elasticity")
        plt.xlabel("Price (RM)")
        plt.ylabel("Predicted Daily Demand (Units)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"需求弹性曲线图已保存至: {save_path}")

    def run(self) -> None:
        """一键执行：生成数据 → 拟合 → 写回 config.json → 绘图。"""
        self.generate_data()
        self.fit_elasticity()
        self.export_config()
        self.plot_curve()


if __name__ == "__main__":
    ShopeeDemandPipeline().run()
