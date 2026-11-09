# Caging Vision-Language Navigation for Legged Robots via Barrier-Certified Safety Architecture（arXiv:2611.04972）

> 来源归档（paper）

- **标题：** Caging Vision-Language Navigation for Legged Robots via Barrier-Certified Safety Architecture
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.04972>（v1，2026-11-03，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.04972>
- **作者 / 机构（论文首页）：** Abhijeet Mangesh Kulkarni、Saimouli Katragadda、Yuze Du、Guoquan Huang（特拉华大学 Robot Perception and Navigation Group，RPNG）；Zijun Sha（丰田汽车公司 Frontier Research Center，东京）
- **发表信息：** 首页标注 Accepted to IEEE-RAS Humanoids 2026
- **代码：** 论文正文未给出代码或项目页链接（本次只读了论文 PDF，未另行检索）
- **入库日期：** 2026-11-09
- **核对范围：** 下载 arXiv v1 PDF（pdftotext 抽取文本），逐段读完摘要、引言、相关工作、系统概览（III）、高层 VLA 规划（IV）、物理接地层与定理 1（V）、实验（VI-A 至 VI-G，含表 I 至表 V 全部数值，已自行核算表 I、II 的聚合行）、结论、附录 I 的定理证明与参考文献列表。图 1 至图 4 只读了图注和正文对图的描述，没有逐图检视像素内容。论文无其他附录或补充材料。
- **通读：** 已完成
- **一句话说明：** 把预训练 VLM/VLA 的离散导航动作当作「提议」，在其与腿式机器人运动控制器之间插入一层「笼」：增量 SDF 建图 + 贝塞尔参数化 MPPI + 离散指数屏障函数（dEBF）硬滤波与一步安全备用策略，在平面刚体模型等假设下给出碰撞规避的前向不变性证明，并在 Unitree G1 与 Ghost Robotics Vision 60 实机验证。

## 核心摘录

1. **问题与定位（摘要、I、II）：** VLA/VLM 的导航输出不显式包含机器人轮廓、局部障碍几何和形式化闭环安全条件；作者不改动语义模型，而在其输出与运动控制器之间加物理接地层（称 caging）。相关工作对比 NaVILA、QUAR-VLA（学习式前端），SafeVLA、VLSA/AEGIS（操作侧安全），Splat-Nav，以及 Shield-MPPI、BR-MPPI、MPPI-DBaS 等 MPPI+屏障函数方法（II）。
2. **语义层（III-B、IV）：** VLM 预测离散动作，如每次左/右转 15°（上限 45°）、前进 25 cm 增量（上限 75 cm）；Kinematic Goal Projector 把动作映射为平面目标位姿，要求 SDF 值 ϕ(p) ≥ d_clear = 0.3 m，被占用时在局部半径内采样替代点，按式 (2) 的代价选最优。任务级动作序列完成后可启用 Grounding-DINO + SAM2 做开放词表目标接近，目标距离需落在 0.5 至 3.0 m，停止距离 d_stop = 0.50 m。
3. **物理层（III-B、V）：** 平面位姿 [px, py, ψ]、机体系速度 [vx, vy, ω] 的恒速离散模型 (1)；人形用 FastLIVO2、四足用不变 EKF（腿运动学 + IMU）做状态估计，增量 SDF 用 Fiesta 类方法（参考文献 [27]）；MPPI 的控制序列用度 d 的贝塞尔曲线参数化 (3)，代价 (4) 至 (8)；机器人用 M 个体固定球近似，h_i = ϕ(p_i) − r_i，dEBF 条件 h(x_{k+1}) ≥ γ h(x_k)（式 10，γ ∈ [0,1)）。选择排序后第一个满足 (10) 的 rollout 的首个指令，若没有则执行一步安全备用 κ(x_k)（式 12，算法 1）。
4. **理论（V、附录 I）：** 定理 1 在 (A1) 恒速模型、(A2) 几何与 SDF 有效、(A3) 备用策略存在、(A4) 只执行通过 (10) 的指令、(A5) x0 ∈ S 下给出 h_i(x_k) ≥ γ^k h_i(x_0) ≥ 0，即安全集前向不变；证明为对 k 的归纳（附录 I）。论文说明在模型 (1) 下零速度是有效备用。
5. **实验设置（VI-A）：** γ = 0.95；MPPI 每次更新 200 样本、10 次迭代，步长 0.05 s、50 步，预测时域 2.5 s；计算平台 NVIDIA A4500；前端为 ChatGPT、Gemini、Claude、NaVILA 四个；指标 SR、CR、E2E SR、Object SR、NE。
6. **受控消融（VI-C，表 I/II）：** G1 在 4 m x 4 m 场地，每个前端 5 次、每种配置聚合 20 次。无 JPS 且静态障碍：CR 80% -> 0%，Object SR 35% -> 40%，E2E SR 15% -> 40%（表 I）。启用 JPS 且中途突现障碍：CR 100% -> 0%，Object SR 55% -> 70%，E2E SR 0% -> 70%（表 II）。已核算聚合行与分项一致。表 I 中 Claude 前端在 dEBF 开启后 Object SR 80% -> 0%、E2E SR 40% -> 0%，论文正文承认 Claude 无碰撞但未完成接近任务。
7. **端到端与跨平台（VI-D，表 III）：** G1 在 6 m x 6 m 静态杂乱环境，25 条指令各 3 次，每种配置 75 次；CR 简单任务 0.33 -> 0.00，复杂任务 0.50 -> 0.00，SR 均为 0.75 不变。Vision 60 只有类别级比率：简单任务 SR 0.47 -> 0.73、CR 0.67 -> 0.20；复杂任务 SR 0.46 -> 0.54、CR 0.91 -> 0.08。剩余 Vision 60 碰撞主要来自细电线、桌沿、窄杆和旧款 VLP16 激光雷达盲区。
8. **系统级笼消融（VI-E，表 IV）：** G1 在 12 m x 12 m 室内，15 条指令 x 4 个前端 x 开/关笼 = 120 次执行；无笼 60/60 碰撞，有笼 0/60。笼同时启用 MPPI、JPS、dEBF，无笼三者全关。前端名称按论文表 IV 原样记为 ChatGPT 5.6 Luna、Gemini 3.5 Flash、Claude Fable 5、NaVILA。
9. **目标接近（VI-F，表 V）：** Vision 60 上平均终点误差 NE 由 1.50 m（关，5 个终点）降至 0.29 m（开，3 个终点）。
10. **论文自述局限（VI-G、VII）：** 认证基于当前 SDF，无法覆盖未观测几何；平面刚体与多球模型不含状态估计误差、延迟和跟踪误差；笼无法纠正错误或不可达的语义目标；未来工作含状态估计与建图不确定性、运动障碍物、抓取与交接等技能、多模态记忆。

## 与周报说法的核对（具身运控lab 周报，条目 01）

- 机构「特拉华大学、丰田汽车前沿研究中心」：与首页一致（特拉华大学 RPNG；丰田汽车公司 Frontier Research Center，东京）。论文致谢另列 Toyota Motor Engineering & Manufacturing North America 资助。
- 「VLA 高层导航 + 状态估计、三维环境建模、SDF、贝塞尔 MPPI + dEBF」：与方法一致，但要注意论文的规划层是平面 2D 位姿与 2D SDF（式 1、ϕ: R² -> R），「三维」主要指感知建图输入；高层前端实际输出的是离散转向 / 前进动作，并非连续动作的端到端 VLA。
- 「在给定假设下提供碰撞规避认证」：与定理 1 一致；假设见上文第 4 条。
- 「在人形和四足平台上验证」：属实，但受控消融和大规模研究主要在 G1，Vision 60 仅有类别级比率和 5/3 个终点的 NE（表 III、V）。

## 对 wiki 的映射

- [paper-legged-vln-barrier-caging](../../wiki/entities/paper-legged-vln-barrier-caging.md)
