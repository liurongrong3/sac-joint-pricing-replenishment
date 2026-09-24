# SAC Joint Pricing & Replenishment

单 SKU、单仓跨境电商仿真：用 **SAC** 联合决策 **每日售价 × 补货量**，并与 **ROP+跟价 / 滚动时域 MILP** 在同一 Gym 环境、同一零售 P&L 口径下对照。

提前期 $L=14$、Episode=90 天；需求为相对竞品价的幂律弹性 $D \propto (p/p_{\mathrm{comp}})^{-\gamma}$。价格敏感时，经典报童 / 纯补货 OR 不够用，因此用 RL 做联合决策，用规则与 MILP 做可解释基线。

---

## 环境依赖

```bash
# 建议使用独立虚拟环境
pip install numpy pandas matplotlib torch gymnasium tensorboard pulp
```


| 包             | 用途                                |
| ------------- | --------------------------------- |
| `gymnasium`   | 仿真环境接口                            |
| `torch`       | SAC Actor / Critic                |
| `pulp`        | 滚动 MILP（CBC）；未安装则 MILP 基线自动回退 ROP |
| `tensorboard` | 训练曲线                              |


统一配置文件：[config.json](config.json)（环境、约束、课程难度、训练超参、对照实验参数均在此修改）。

---



## 快速开始



### 1. 仿真搭建（生成合成数据并拟合需求弹性）

```bash
python demand_pipeline.py
```

作用：

1. 按 `config.json → simulation` 生成相对竞品价的合成销量 CSV
2. 拟合幂律参数 $A,\gamma$，写回 `config.json → environment.fit_A / fit_gamma`
3. 输出弹性曲线图（默认 `demand_elasticity_curve.png`）

若已有可用的 `fit_A / fit_gamma`，可跳过本步直接训练。

### 2. 训练 SAC（课程学习 easy → medium → hard）

```bash
python train.py
```

- 读取 `config.json → training / sac_agent / reward_shaping / paths`  
- 预热随机探索 → 课程晋级（滑动窗口平均 P&L 达阈值则升难）  
- 权重写入 `models/`，例如 `sac_actor_best_easy.pth`  
- TensorBoard 日志：`runs/SAC_Shopee_<timestamp>/`

```bash
tensorboard --logdir runs
```



### 3. 基线对照（Random / ROP / MILP / SAC）

```bash
python compare_baselines.py
```

参数全部在 `config.json → baseline_comparison`（及 `evaluation.difficulty`）。  
输出：终端对比表 + `baseline_comparison.csv`。

> **注意：** 当前解释器必须已 `pip install pulp`，否则 MILP 会整局回退成 ROP，两行数字会完全一样。



### 4. 单局可视化评估

```bash
python eval_and_visualize.py
```

加载 `evaluation` 指定难度的 best/final 权重，跑一局并画出库存 / 定价 / 订货等决策曲线。

---



## 项目结构

```
sac_shopee/
├── config.json              # 统一配置
├── config_loader.py         # 配置读取
├── demand_pipeline.py       # 合成数据 + 弹性拟合（仿真搭建）
├── shopee_env.py            # Gymnasium 环境（状态 / 动作 / P&L / 约束）
├── sac_agent.py             # SAC 实现
├── train.py                 # 课程学习训练
├── baselines.py             # Random、ROP+跟价、SAC 封装
├── rolling_milp.py          # 规则跟价 + 滚动 MILP 补货
├── compare_baselines.py     # 同环境多策略对照表
├── eval_and_visualize.py    # 单局评估与曲线
└── models/                  # 训练权重
```

---



## 仿真环境设计


| 设定       | 取值（默认）                                              |
| -------- | --------------------------------------------------- |
| SKU / 仓  | 单 SKU、单仓                                            |
| 提前期 L    | 14 天（在途队列）                                          |
| Episode  | 90 天                                                |
| 单位到岸成本 c | COGS 12 + 头程 3 = **15 RM**                          |
| 固定下单费    | 50 RM / 次                                           |
| 仓储       | ≤200：0.05/件/天；超出部分 0.25/件/天                         |
| 缺货罚      | 15 RM / 件                                           |
| 期末残值     | (现货 + 在途) × $c$ × $\rho$，默认 $\rho=1$ |


**需求模型**

$$
\mathbb{E}[D] = A \cdot \left(\frac{p}{p_{\mathrm{comp}}}\right)^{-\gamma} \times \text{日倍率}
$$

再乘对数正态噪声。课程三档：

- **easy**：低噪声、竞品价固定  
- **medium**：周末流量 + 竞品小幅抖动  
- **hard**：月末大促窗、竞品打折、需求倍率抬高

**库存动态（每日）**：先到货入账 → 映射动作（定价/订货）→ 产生需求与销售 → 结算 P&L → 新订单进入在途队尾（L 天后到）。

---



## 状态空间

观测维度：1 + L + 1 + 1 = 17（L=14）


| 分量                 | 含义             |
| ------------------ | -------------- |
| `inventory`        | 当日现货           |
| `pipeline[0..L-1]` | 在途队列（队头为下一步到货） |
| `comp_price`       | 当日竞品价          |
| `promo_flag`       | 是否平台大促窗（0/1）   |


动作空间：`Box([-1,1]^2)`


| 分量 | 映射 |
| --- | --- |
| $a_p$ | 线性映射到 [`min_price`, `max_price`]，再经约束夹紧 |
| $a_q$ | 线性映射到 `[0, max_order_qty]`，低于起订阈值视为 0 |


---



## 奖励函数设计

评估与报表用真实 **净利** `net_profit`；SAC 优化的是带 shaping 的 `reward`（二者刻意可分离）。

**真实 P&L（评估）**

$$
\mathrm{net\_profit} = (p\cdot s - c\cdot s - F\cdot \mathbf{1}_{q>0}) - S_{\mathrm{storage}} - b\cdot o - \mathrm{opening} + \mathrm{salvage}
$$

- $s$：当日销量；$o$：缺货件数；$F=50$：有订货才扣  
- 采购变动成本在**售出日**结转，不在下单日一次性扣完  
- 期初扣减 / 期末残值配对，避免「白捡开局库存」

**训练 reward（默认权重见 `reward_shaping`）**

$$
r = p\cdot s - c\cdot s - F\cdot \mathbf{1}_{q>0} - w_{\mathrm{storage}} S_{\mathrm{storage}} - w_{\mathrm{stockout}}(b\cdot o) - w_{\mathrm{price}}\left(\frac{|p-p_{\mathrm{comp}}|}{p_{\mathrm{comp}}}\right)^{2} - \mathrm{opening} + \mathrm{salvage}
$$


默认：`w_storage=1.2`，`w_stockout=1.1`，`w_price_extreme=80`。  
写入 Replay Buffer 前再乘 `reward_scale=1e-4`，防止 Q 值爆炸；日志里的 P&L **不缩放**。

---



## 约束处理

约束在环境 `_map_actions` / `step` 中执行（硬投影），不依赖网络自己学合规。


| 约束 | 做法 |
| --- | --- |
| 成本地板 | $p \ge c + \texttt{min\_margin}$（默认 +5 RM） |
| 竞品价带 | $p$ 夹在竞品价 $\pm 45\%$ 与业务价上下限的交集内 |
| 单次订货上限 | $q \le \texttt{max\_order\_qty}=200$ |
| 最小起订 | $q < 10$ → 订 0（避免天天付固定费） |
| 库存头寸帽 | 现货 + 在途 + $q$ ≤ 650；超限则当日 $q=0$ 或截断 |
| 到货后再截断 | `step` 内先 `popleft` 到货，再按剩余头寸夹一次 $q$ |


规则 / MILP 基线共用同一套跟价与头寸逻辑，保证对照公平。

---



## 对照策略说明


| 策略 | 定价 | 补货 |
| --- | --- | --- |
| Random | 随机 $[-1,1]^2$ | 同左 |
| ROP+MatchPrice | 竞品价 × `match_ratio`（默认 0.98） | $(s,S)$：`ROP=μL+SS`，`SS=safety_days·μ` |
| MILP+MatchPrice | 同上跟价（价格**不进**求解器） | 滚动窗口 $N=21$，PuLP+CBC，**只执行当天 $q_0$**；无解回退 ROP |
| SAC | 联合学 $p,q$ | 联合学 $p,q$ |


MILP 为何不联合定价：收入 $p\cdot\min(I,D(p))$ 含幂律非线性与 $p \times s$ 双线性，不再是 MILP。求解器设 `timeLimit=5s`、`gapRel=0.01`。

---



## 典型结果

装好 PuLP 并训练出 `models/sac_actor_best_easy.pth` 后，用 `compare_baselines.py` 得到均值±标准差（easy，5 episodes，seeds 42–46）：

| 策略 | 净P&L均值(RM) | P&L±std | 缺货天数 | 平均库存 | 仓储费 | 缺货罚金 | 均价(RM) | 最大跟跌% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 30,232.2 | 7,251.6 | 43.0 | 40.2 | 206.9 | 30,645.0 | 37.85 | 42.9% |
| ROP+MatchPrice | 49,255.9 | 218.6 | 45.8 | 34.7 | 357.4 | 20,052.0 | 34.30 | 2.0% |
| MILP+MatchPrice | 47,522.6 | 417.2 | 44.8 | 17.6 | 79.0 | 21,537.0 | 34.30 | 2.0% |
| SAC | 73,651.3 | 255.9 | 12.0 | 148.7 | 839.9 | 5,505.0 | 44.51 | 0.0% |

读表时注意：

- 跟价固定时，**MILP 与 ROP 利润接近**（都只优化补货；本表 MILP 库存更瘦、仓储更低，缺货罚略高）  
- **SAC** 利润更高、缺货更少，代价是平均库存和仓储上升——均价约 44.5，优势主要来自**联合定价**



## 下一步计划：多 SKU

单品只是把账算清楚。店里一多，真正难的是：**钱和仓都只有一份，给谁补、给谁推都要取舍。**

例如：总预算和仓位怎么分；一次下单的 50 块杂费要不要几个 SKU 拼在一起；A 降价会不会把 B 的客人抢走；大促主推引流款还是利润款。

我计划按这个顺序扩，做一步算一步：

1. 各 SKU 先仍跟价 + 再订货点，上面加一层按风险/毛利切预算和仓位。  
2. 再上多品滚动运筹：价格先钉死，一次算出各品当天订多少；算不动就缩小窗口或拆开解，解不出就退回规则。  
3. 有店内数据再考虑替代、抢流量。

应该不会一上来就用 MARL。

多 SKU 是后续方向，不是本仓库已经做完的部分。

