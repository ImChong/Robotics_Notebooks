# Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control（arXiv:1909.06586）

> 来源归档（原文通读 · 支撑 `wiki/overview/humanoid-motion-control-know-how.md` 的 WBC / 摩擦锥 / 多速率条目）

- **标题：** Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control
- **作者：** Donghyun Kim, Jared Di Carlo, Benjamin Katz, Gerardo Bledt, Sangbae Kim（MIT）
- **类型：** paper / quadruped / WBC / MPC
- **原文：** <https://arxiv.org/abs/1909.06586>（arXiv:1909.06586v1，2019-09-14；2026-11-07 获取 PDF，pdftotext 提取全文）
- **入库日期：** 2026-11-07
- **一句话说明：** 用 MPC（简化模型，低频）算足端反力轨迹，WBIC（全身脉冲控制，高频）把反力与姿态 / 摆腿任务转成关节力矩、位置、速度指令，在 Mini-Cheetah 上做到 3.7 m/s。
- **核对范围：** 通读摘要、§I 引言、§II 系统概览、§III MPC（式 6–11，含摩擦锥线性近似、步态调度、落脚点规划）、§IV WBIC（任务优先级的零空间迭代式 16–24、QP 式 25、力矩式 26）、§V 实验（任务与增益表 I、高速奔跑、关节速度 / 力矩 / 功率）、结论；参考文献仅浏览；视频未读，未运行代码（该论文未随文给出代码）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **多速率架构（Fig. 1–2）：** 状态估计、WBIC、运动学 / 动力学模型 0.5 kHz；MPC 0.04 kHz（引言与图中写 40 Hz，另一处写 30 Hz，以原文为准存在不一致）；关节级 PD 40 kHz，约为高层全身控制的 80 倍。输入为编码器的关节位置 / 速度与 IMU 的角速度 / 线加速度。
- **状态估计：** 基于运动学与加速度数据的卡尔曼滤波器估计全局位置与速度；作者称其对性能贡献显著，但未在文中展开。
- **摩擦锥（式 11）：** MPC 把摩擦锥近似为四棱锥形不等式 $|f_x|\le\mu f_z,\ |f_y|\le\mu f_z,\ f_z>0$；论文正文未给出 $\mu$ 的具体取值。
- **WBIC 增益（表 I）：** 体姿态 / 体位置 / 足位置三类任务的反馈增益 $k_p=100\,\mathrm{s^{-2}},\ k_d=10\,\mathrm{s^{-1}}$；关节级反馈统一 $k_p=3$ N·m/rad、$k_d=0.3$ N·m·s/rad（髋外展关节 $k_d=1$）；QP 中反力权重 $Q_1=1$、浮基权重 $Q_2=0.1$。
- **为何用算子空间而非关节阻抗：** Mini-Cheetah 的本体感知（可反驱）执行器通过电机电流直接开环控力矩，故直接控算子空间阻抗比依赖关节阻抗控制更有效（后者会延迟算子空间控制）。
- **硬件利用：** 最高观测速度 4 m/s（随后失稳），以 3.7 m/s 为记录；髋关节最大速度 34 rad/s、指令力矩 25.5 N·m，而额定最大 40 rad/s、17 N·m——图中力矩是发给电机控制器的**指令**，超过上限时执行器输出会被最大力矩截断；峰值功率 280 W 大于规格 250 W。
- **作者未写的内容：** 论文没有讨论电池压降、电机温升、IMU 减振或奇异点保护等工程问题。
