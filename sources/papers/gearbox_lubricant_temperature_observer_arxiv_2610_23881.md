# Gearbox Lubricant Temperature Observer（arXiv:2610.23881）

> 来源归档（paper）

- **标题：** Lubricant Temperature Observer for Gearboxes in Industrial Robots
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.23881>
- **PDF：** <https://arxiv.org/pdf/2610.23881>
- **项目页：** <https://doi.org/10.18419/DARUS-5015>
- **入库日期：** 2026-11-01（2026-11-21 完成 `arXiv:2610.23881v1` 全文通读与参数/验证表核对）
- **通读：** 已完成
- **线索来源：** [模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](../blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)（具身智能小站，2026-11-01）
- **一句话说明：** 针对工业机器人摆线齿轮箱润滑脂温度内部难直测的问题，基于两状态热传递模型构建 Luenberger 观测器与 Kalman 滤波器（仅测壳体与环境温度），结合温度相关 Stribeck 摩擦模型将润滑温度平均绝对误差压至 **1.3 K**（供滞回与传动误差补偿）。

## 开源状态

- **项目页未列 GitHub**（步骤 2.5 核查，2026-11-01）：项目页与正文 Data Availability（<https://doi.org/10.18419/DARUS-5015>）仅发布实验测量数据集，未附 MATLAB/Simulink 观测器代码仓库。

## 核心摘录（原文 PDF 精读实证）

- **实验台架（§II-A）**：**Nabtesco RH380-N** 摆线减速器（减速比 **185:1**、额定力矩 **3724 N·m**），使用非牛顿 **Nabtesco RV GREASE LB00** 润滑脂（密度 $0.87\,\text{g/cm}^3$，25–50 °C 与 10–1000 1/s 剪切率下运动粘度跨 **200–9000 mPa·s**）；由 **Bosch Rexroth MSK070D**（额定 17.5 N·m）驱动，3 只 PT100 经 Beckhoff EL3201-0010 采集（0.01 K 分辨率、<±0.1 K 不确定度）。
- **两状态热模型与辨识参数（§II-B, Table I）**：以摩擦热功率 $P_{L,\text{in}}=\tau_f\dot{\theta}$ 与环境温度 $T_E$ 为输入，状态 $x=[T_L, T_H]^\top$、输出 $y=T_H$；经 5 组恒速（0.6–1.0×90°/s）积分最小二乘辨识得均值为 $C_L=168.6251\,\text{J/K}$、$C_H=95.8339\,\text{J/K}$、$K_{LH}=0.2132\,\text{W/K}$、$K_{HE}=0.0661\,\text{W/K}$；摩擦采用二次温度多项式系数的平滑 tanh-Stribeck 模型（LM 算法辨识）。
- **观测器参数与验证指标（§III, Table II）**：Luenberger 增益 $L=[0.1675, 0.0368]^\top\,\text{s}^{-1}$，Kalman 过程噪声 $Q=\text{diag}(0.4, 0.01)\,\text{K}^2$、量测噪声 $R=7.2855\times10^{-5}\,\text{K}^2$；跨 5 条升温、2 条 4 h 长周期及 12 h 冷却验证轨：纯模型 MAE **0.8427 K**（最大误差 3.4389 K），Luenberger MAE **1.3073 K**（最大 3.6098 K），Kalman MAE **1.2956 K**（最大 3.8232 K）；当初始油温错设 30 °C（实际 40 °C）时，纯模型 13 min 后仍偏差 −5 K，两观测器均在约 **3.2 min** 内快速收敛（13 min 后误差仅 3.1 / 3.4 K，满足 <5 K 误差补偿需求）。
- **局限（§IV）**：静置阶段油脂因不再循环且粘度升高产生底部积聚与剪切率相关热导下降，导致停机初始阶段实测油温骤降、模型与观测器皆偏离真实静置物理过程；且本次验证仅覆盖 0 N·m 空载工况。

## 对 wiki 的映射

- [paper-gearbox-lubricant-temperature-observer](../../wiki/entities/paper-gearbox-lubricant-temperature-observer.md)
- [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)
