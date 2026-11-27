# Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping

- **作者：** William Liang, Sam Wang, Hung-Ju Wang, Osbert Bastani, Yecheng Jason Ma, Dinesh Jayaraman
- **机构：** University of Pennsylvania、University of California, Berkeley、Dyna Robotics
- **版本 / 载体：** arXiv:2603.03278v1 [cs.RO]（2026-03-03）；ICLR 2026
- **原文：** https://arxiv.org/abs/2603.03278 、https://arxiv.org/html/2603.03278 、https://tether-research.github.io
- **项目页 / 代码：** 已开源 — 项目页 https://tether-research.github.io ；官方仓库 https://github.com/tether-research/tether （MIT 协议）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections 1–5、References、Appendix A.1–A.5 含 Algorithm 1–2、Listings 1–4、Tables 1–3）及项目页快照文本，并与官方仓库 `tether-research/tether` README 及文件清单核对
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 针对机器人模仿学习依赖大量人工遥操作示教、数据难以扩展的问题，提出 Tether：将少样本高鲁棒性的开环轨迹扭曲策略（Trajectory Warping）与 VLM 引导的多任务自主功能性玩耍（Autonomous Functional Play）结合，在真实世界低人工干预下持续生成大规模专家级交互轨迹用于下游闭环策略训练。
- **核心技术（Section 3 & Appendix A.1–A.2）：**
  1. **演示摘要（Demonstration Summaries）：** 将每条演示 $\tau_i$ 预处理为 $\kappa_i = (o, W, K, \mathbf{a})$，其中 $o$ 为左右双目第三视角初始图像，$W = [w_1, \dots, w_T]$ 为夹爪开合切换时刻提取的 3D 关键路点（亦支持在 Appendix A.1 中通过 VLM 选关键帧并标注 2D 像素三角化获取，如倒水任务），$K$ 为 $W$ 投影至图像平面的 2D 关键点，$\mathbf{a} = [a_1, \dots, a_M]$ 为完整 6-DOF 夹爪位姿与开合动作序列。
  2. **语义关键点对应与源演示选择：** 使用基于 DINOv2 与 Stable Diffusion 特征的 GeoAware 语义对应模型（Zhang et al., 2024）在当前左右视图 $o$ 中匹配每个演示关键点 $K_i$，经相机外参反投影三角化得到目标 3D 路点 $\tilde{W}_i$。通过双约束过滤无效匹配：（a）左右反投影射线交汇误差 $>10\text{ cm}$ 视为不可行；（b）利用 MASt3R（Leroy et al., 2024）进行跨视角点匹配，校验 3D 路点到对应视图射线的距离差是否超过 $10\text{ cm}$。可行演示按 $\text{score}_i(o) = \|W_i - \tilde{W}_i(o)\|_2$ 排序选取最近邻 $\kappa^*$。
  3. **空间插值轨迹扭曲与速度重采样：** 对相邻路点段 $[w_t, w_{t+1}]$ 计算端点位移 $d_t = \tilde{w}_t - w_t$ 与 $d_{t+1} = \tilde{w}_{t+1} - w_{t+1}$，沿局部 1D 空间投影坐标 $\alpha = (a - w_t^*)^\top v / \|v\|^2$ 对位移做空间线性插值 $d_a = (1-\alpha)d_t + \alpha d_{t+1}$，得到 $\tilde{a} = a + d_a$；随后按新旧段弧长比例 $\tilde{T}_{\text{segment}} = (\tilde{L}/L)T_{\text{segment}}$ 重采样时间步长以保持执行速度一致。
  4. **VLM 引导的多任务自主玩耍：** 构建近似无限可组合（approximately indefinitely composable）的 6 任务集合（桌面/货架/碗之间的菠萝与碗搬运），每轮以 $\text{Softmax}(-|\mathcal{G}_t|)$ 优先采样成功数少的稀缺目标任务，调用 $\text{VLM}_{\text{planner}}$（Gemini Robotics-ER 1.5）生成达到目标的可行动作序列并执行首步任务（类似滚动时域控制），再用 UCB 多臂赌博机从任务演示库筛选 $k$ 条高质量源演示交由 Tether 执行，最后由 $\text{VLM}_{\text{evaluator}}$ 对比执行前后多视角（左、右、腕部）图像判定成功并将成功轨迹入库 $\mathcal{G}_t$，同时更新 UCB 奖励。
- **关键实验数据（Section 4 & Appendix A.4–A.5）：**
  - 硬件平台：7-DOF Franka Emika Panda（15 Hz 控制），双 ZED 第三视角相机（评价时另附腕部相机），1× NVIDIA A6000 GPU。
  - **Table 1（12 项操作任务，每项 10 次试验，均提供 $\le 10$ 条演示）：**
    - 分布内空间泛化（Pineapple from Shelf / Pineapple to Bowl / Bowl to Shelf / Bowl from Shelf）：Tether (10 Demos) 分别为 `1.0 ± 0.0`、`1.0 ± 0.0`、`0.9 ± 0.09`、`1.0 ± 0.0`；1 Demo 下仍达 `0.8`、`0.9`、`0.7`、`1.0`；零样本 $\pi_0$-FAST 为 `0.5`、`0.7`、`0.0`、`0.4`；10-demo Diffusion Policy 为 `0.0`、`0.0`、`0.0`、`0.1`；微调 $\pi_0$ 与 KAT 在这批任务上均为 `0.0`。
    - 分布外语义泛化（Apple to Bowl / Strawberry to Bowl / Pineapple to Basket / Pineapple to Cup）：Tether (10 Demos) 均达 `0.9 ± 0.09`；零样本 $\pi_0$ 为 `0.6`、`0.7`、`0.4`、`0.1`；DP 为 `0.0`。
    - 高难度接触/形变/铰接/精密任务（Wiping with Cloth / Opening Cabinet / Hanging Tape / Inserting Coffee，其中咖啡胶囊插入误差容限 8 mm 需拉近相机）：Tether (10 Demos) 分别为 `0.6 ± 0.15`、`0.6 ± 0.15`、`0.7 ± 0.14`、`0.4 ± 0.15`；$\pi_0$ 与 DP 均为 `0.0`。
  - **Table 2（与关键点模仿策略 KALM 对比）：** 在 4 项菠萝/碗任务上 Tether 为 `1.0 / 1.0 / 0.9 / 1.0`，KALM 为 `0.6 / 0.0 / 0.0 / 0.5`。
  - **自主玩耍（Section 4.3）：** 4 轮累计约 26 小时、1946 次尝试、生成 1085 条成功轨迹（平均每 48 秒尝试 1 次、每 86 秒成功 1 次，累计成功率 55.8%）；仅需 5 次人工干预（占比 0.26%，约每 5.2 小时一次，总计约 1 分钟人工时间；其中 4 次为碗完全倒扣、1 次为菠萝被推入货架深处）。VLM 规划准确率 95.2%，成功评估精确率 98.4%（召回率 89.6%）。
  - **下游策略学习（Figure 8–9 & Table 3）：** 用过滤行为克隆（Filtered BC）训练每任务 Diffusion Policy 与 $\pi_0$。$\pi_0$ 在 Pineapple from Shelf / Pineapple to Bowl / Bowl to Shelf 上从 10 条人工演示的 `0.0 / 0.0 / 0.0` 随 play 数据增加至全部数据时提升到 `0.7 / 0.8 / 0.4`（高于预训练零样本的 `0.5 / 0.7 / 0.0`）。在玩耍诱导分布（20 个采样状态）上，10 条演示的 Tether 仍显著优于用 141–202 条人工演示训练的 Diffusion Policy。
- **对 wiki 的映射：** [Tether 知识实体页](../../wiki/entities/paper-agentic-2603-03278.md)
