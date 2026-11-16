# METATRON（arXiv:2610.38027）

> 来源归档（paper）

- **标题：** METATRON: A Promptable Whole-Body Execution Layer for Humanoids
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.38027>
- **PDF：** <https://arxiv.org/pdf/2610.38027>
- **作者：** Tessler, Chen, Jiang, Yifeng, Otto, Alec, Shi, Yi, Iqbal, Umar, Chechik, Gal, Peng, Xue Bin
- **arXiv 评论栏：** （无）
- **入库日期：** 2026-11-02
- **通读：** 已完成
- **核对范围：** arXiv:2610.38027v1 正文（§I–§VII，含 Fig. 2–3、eq. (1)–(6)、Table I–II、§VI-F 消融与 §VII 局限）；注：附录与 Table III 细分行未逐表展开。
- **一句话说明：** METATRON：可提示的人形全身动作执行层，将生成式运动执行基础策略与可编程有状态技能层解耦。

## 开源状态

- **代码与模型已开源：** 框架代码合并于 `NVlabs/ProtoMotions`（Apache-2.0），Unitree G1 检查点及导出 ONNX 发布于 `huggingface.co/nvidia/Metatron-v1`（NVIDIA Open Model License）；SOMA 预训练权重、浏览器演示 TypeScript 前端源码与原始训练动作数据集未随模型卡分发。

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 不同人形任务通常分别训练控制策略，重复学习平衡和全身协调能力。
3. **周报所述创新点：** 冻结一个接收身体部位目标、速度和风格提示的生成式基础策略；高层技能以有状态程序逐周期生成提示，使编码智能体能够通过修改程序而不是重新训练策略来构建新技能。
4. **arXiv 摘要（英文节选）：** Humanoid controllers often require task-specific training despite sharing the need for balance and whole-body coordination. We present Metatron, a promptable whole-body execution layer combining a generative base policy with a programmable skill layer. The base policy receives partial goals, collectively called a prompt, that can combine timed body-part targets, velocity commands, and style cues.

## 全文精读与核验笔记（2026-11-14）

### 1. 两阶段基础策略训练（§III-A/C/D）

- **Stage 1（特权运动跟踪专家）：** 在仿真中使用 PPO + FSQ 稠密参考动作跟踪奖励训练 4 个领域专家教师策略（`BONES`、`AMASS+LAFAN`、`recovery` 恢复、以及面向家具交互的 `SAMP`）。跟踪器读取特权仿真状态与完整未来参考轨迹（eq. 1–2）。
- **Stage 2（掩码条件蒸馏）：** 将教师动作蒸馏到单个 MaskedMimic 风格的生成式学生策略（G1 与 SOMA 各有一个独立检查点）。训练混合采用 posterior 引导的学生 rollout 与冻结教师 rollout（eq. 5），同时对输入 prompt 进行随机掩码以支持部分目标提示。

### 2. Prompt 接口与球面条件 VAE（§III-B/C）

- **Prompt 结构 $g_t = (C^w_t, C^\ell_t, u_t, \xi_t, b_t)$：** 世界系时空身体部位位置/朝向目标 $C^w_t$、躯干规范化局部姿态目标 $C^\ell_t$、航向系平面速度指令 $u_t$、无序风格样本池 $\xi_t$（编码为隐式风格条件）、以及场景障碍物 8 角点包围盒 $b_t$。
- **Prior / Posterior 与单位超球面隐空间：** Prior Transformer 读取可部署本体/动作历史 $(o_t, a_{t-1})$ 与被掩码的 $g_t$，输出确定性条件特征 $d_t$ 及高斯参数 $(\mu_t, \sigma_t)$；训练态 Posterior 额外读取特权状态 $(s_t, y_t)$ 预测残差 $(\mu_t+\Delta\mu_t, \sigma'_t)$。采样结果经 L2 归一化投影到单位超球面 $z_t = \tilde{z}_t/\|\tilde{z}_t\|_2$，与 $d_t$ 经融合层 $h_t = f_\theta(d_t, z_t)$ 解码出关节目标 $a_t$。
- **辅助正则化：** 除 KL 散度与 $\mu_t$ 单位范数正则外，引入下一步条件预测 $F(h^q_t) \to \mathrm{sg}(d_{t+1})$、相邻时序特征一致性及动作平滑损失。真机部署仅运行确定性 prior 均值路径 + decoder，不执行 posterior。

### 3. 可编程技能层与数据工程（§IV–§V）

- **技能层范式：** 技能是每个控制周期更新结构化 prompt 的有状态普通程序（非控制环内跑 LLM）。人类给出初始规则模板，编码智能体（coding agents）分析动捕切片并在仿真评测场中迭代修改技能代码，全程冻结基础策略权重：
  - **步态转向：** 利用周期函数拟合脚部目标轨迹 $p_i(\tau)$（eq. 6）；
  - **坐下技能：** 给定座椅中心上方/后方躯干目标与座椅前缘双脚目标的三点约束；
  - **跌倒起身恢复：** 基于 FK 躯干高度的滞回阈值切换；
  - **交互顺应：** 设定零速度与可控躯干高度/朝向，而将手臂目标置空保持顺应。
- **无碰撞 SAMP 与恢复数据：** SAMP 家具交互采用朝向表面点云与穿模惩罚（惩罚穿透但不惩罚接触）做两阶段无碰撞重定向；此外补充了合成的 HIL 风格恢复轨迹。

### 4. 核心实验与局限（§VI–§VII）

- **实机与仿真平台：** Unitree G1 真机以 C++ TensorRT + CUDA graphs 在笔记本 GPU 上经以太网 `50 Hz`（`kControlDt = 0.02s`）运行单个冻结基础策略，支持 Xbox 手柄与 Pico 遥操作；SOMA 为仿真角色验证。
- **Table I（54 目标全身腕部触达，0.35–1.8 m，无合成恢复数据，10 cm 保持 + 未跌倒）：**
  - METATRON：**83%** 触达率，**5.9 cm** 平均误差，**100%** 保持直立；
  - ARDY+SONIC：39%（6.4 cm，82%）；GRU+IK+SONIC：28%（5.6 cm，81%）；GRU+SONIC：0%（—，87%）。
- **Table II（1 m/s 前进中 100 ms 躯干推扰，8 方向，8 s 未跌倒率）：**
  - METATRON：**100%** (@200N)、**100%** (@400N)、**62%** (@600N)、**75%** (@800N)、**75%** (@1000N)；
  - ARDY+SONIC：100% / 100% / 17% / 0% / 0%；GRU+SONIC：100% / 75% / 25% / 0% / 0%。
- **§VI-F 消融：** 完整模型在 0–200N 推扰下优于 Global-only、Mocap-only 与 No-aux，在所有非零推力下优于适配后的 MaskedMimic+ 与 HOVER+；不过当移除主动迈步技能、仅输入原始位置路点时，所有对比变体在 600–800N 下的站立率均低于 10%。
- **§VII 局限：** 对训练动作覆盖敏感（静止状态突变下达 2 m/s 指令可能触发类似前滚翻的异常过渡，实机用输入 EMA 平滑处理）；不自带感知与高层任务规划，场景包围盒 $b_t$ 与技能切换触发依赖外部系统（如坐沙发实验将机器人预先置于沙发旁触发）。

**对 wiki 的映射**

- [paper-metatron-promptable-whole-body-execution](../../wiki/entities/paper-metatron-promptable-whole-body-execution.md)
