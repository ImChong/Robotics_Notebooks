# Robo-Harness K1: Harnessing Robot-Use Agents via Perception Augmentation

- **作者：** Zexi Li, Yehang Zhang, Wenqian Li, Haojian Huang, Chenxu Wang, Chunyang Zhang, Shiyuan Deng, Yangkai Wei, Tianyi Zhang, Ruize Gao, Binghui Xie, Bohan Zhou, Jinghao Li, Kaiwen Zhou, Yifan Chang, James Cheng, Yinchuan Li
- **机构：** 原文 arXiv HTML 与项目仓 README 未披露作者单位
- **版本 / 载体：** arXiv:2609.29389v2（2026-09-24 提交 v1，2026-11-24 修订 v2）；arXiv HTML 全文
- **原文：** https://arxiv.org/html/2609.29389 （摘要页：https://arxiv.org/abs/2609.29389 ）
- **项目页 / 代码：** https://github.com/Robo-Harness/k1 （项目主页：https://robo-harness.github.io/k1/ ；**部分开源**：以 MIT 协议开源 K1 核心框架、LIBERO-Pro 与 RoboSuite 适配器及 Qwen3.5 LoRA SFT 训练接口，含 `v1.0` 与 `v1.1` 版本；RoboCasa、RoboTwin、RoboDojo、RoboLab 适配器待公开发布，数据集、模型权重与原始轨迹日志未随仓提供）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections 1–5、Appendix A–E，含表 1–6 与表 S1–S26、图 1–8 与图 S1–S5），并核对官方 GitHub 仓库 `Robo-Harness/k1` 主页 README 与文件列表。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心定位

- 现有视觉-语言模型（VLM）能理解 2D 场景语义与指令，但在作为机器人使用智能体（Robot-Use Agent, RUA）直接控制机械臂时，难以从纯 RGB 图像中恢复运动控制所需的 3D 度量几何关系。例如在相同纯 RGB 接口（Inspect Robots）下，GPT-6 Astra 在 LIBERO-PRO 60 组配对样例中达 63.3%，而 Gemini 3.7 Flash 仅 16.7%（Spatial 类仅 5.0%），且失败常表现为高度/深度估计错误或过早宣称完成。
- **Robo-Harness K1** 提出 **perception as tools（感知即工具）** 范式：不修改基础 VLM 架构与输入模态，而是将标定后的空间感知、持续锚点与抓取候选封装为与物理动作同一格式的可调用工具；VLM 在「观测 → 测量 → 动作 → 验证」闭环中按需调用工具获取度量回执与 RGB 叠图证据，并结合自适应运动粒度与记忆检索完成操作。

### 2. 方法体系

- **双工具集闭环（Section 3.1）：** 工具集划分为只读信息查询 $\mathcal{U}_{\rm query}$（返回测量回执 $r$ 与视图叠加，不移动机器人）与物理执行指令 $\mathcal{U}_{\rm act}$（执行运动并返回回执与新观测 $o_{t+1}$）。智能体每次决策基于指令 $g$、当前 RGB 视图与机械臂状态 $o_t$、交互上下文 $c_t$ 采样工具调用 $(u, a) \sim \pi_\theta(\cdot \mid g, o_t, c_t)$。
- **四大感知工具族（Section 3.2，Appendix A.1–A.5）：**
  1. **区域定位（Region Grounding）：** `find_regions` 基于文本提示调用 SAM 3 生成带持久化 `S` 编号的候选掩码叠图；`inspect_region` 支持边框精细分割与局部表面几何汇总；`check_region_view` 将测量表面点投影到另一视角相机验证深度一致性。
  2. **深度与局部几何（Depth & Geometry）：** `measure_depth` 利用相机内参 $K^c$、深度 $D^c(u,v)$ 与外参 $T^w_c$ 将像素反投影至世界坐标 $x^w$ 并返回相对工具中心点（TCP）的位移；`fit_geometry` 拟合可见平面或无向主轴方向；`region_relation` 计算区域间及区域到 TCP 的空间位移关系。
  3. **持久锚点追踪（Anchor Tracking）：** 依托 TAPNext++ 跨帧追踪图像对应点，并结合当前深度与相机位姿提升至世界坐标，从而分离相机运动与表面真实位移；每个锚点维护编号与实时有效性标记（`valid`），并在工具调用之间自动刷新（Section 4.3 指出显式追踪工具 `track_features`/`read_track` 未被主动请求，核心收益来自调用间自动锚点刷新）。
  4. **抓取候选生成（Grasp Candidates）：** `grasp_candidates` 将已定位区域的点云输入 GraspGen，通过具身适配器映射到已标定指垫（finger-pad）中心坐标系，结合双指对称性去重与深度视场检查后返回最多 3 个候选 TCP 位姿、预抓取航点与投影示意图，由模型自主选择并通过通用运动工具执行。
- **动作粒度、记忆检索与完成反馈（Section 3.3，Appendix A.4 / C.2）：**
  - **动作工具：** `move_relative`（每轴局部修正上限 $\delta_{\rm axis}=3\text{ cm}$）、`move_to_pose` / `rotate_toward` / `align_axis`（持久化绝对位姿或单次 $\le 15^\circ$ 姿态对齐）、`move_toward`（长距离目标直达，遇容差/停滞/预算耗尽停止）、`set_gripper`（0 闭 / 1 开，返回夹爪开度与稳定状态）、`retreat` 与 `align_region_references`。
  - **上下文管理：** 默认保留指令、当前证据、最近 $N=8$ 轮文本交互与 $K=1$ 轮历史图像，同时提供 `search_history`、`read_history` 检索完整轨迹与历史图像，以及 `update_task_progress` / `remember` 记录里程碑；每次工具调用附带简短决策注释（decision note）。
  - **完成反馈：** 提交 `done` 时触发环境原生成功判据；若未达标且仍有预算，控制权交回模型继续修正（区别于 RGB-only 基线提交 `done` 即直接终止）。
- **智能体监督微调（Agentic SFT，Section 3.4，Appendix B）：** 将教师模型（Gemini 3.7 Flash）成功轨迹中的工具调用序列化为纯文本目标，过滤执行错误、缺参/多调用及被拒 `done` 后保留 2,482 条目标（来自 107 条成功轨迹，占 2,564 次决策的 96.8%），以语言侧 LoRA（rank 16, $\alpha=32$，冻结视觉编码器）微调 Qwen3.5-9B，无需单独动作头；后续还在 4 环境混合的 316 条轨迹（LIBERO-PRO 107 + RoboSuite 142 + RoboCasa 32 + RoboTwin 35）上继续微调 3 个 epoch。

### 3. 实验与消融核对

- **LIBERO-PRO 前沿模型对比（Section 4.2，表 1、表 S6–S7）：**
  - 60 组配对测试（Spatial/Object/Goal 各 10 任务 × swap/task 变体，initial state 0）：Gemini 3.7 Flash 从 16.7%（10/60）提升至 **71.7%（43/60）**（+55.0 pp，McNemar $p=2.1\times 10^{-9}$），其中 Spatial 从 5.0% 升至 75.0%；GPT-6 Astra 从 63.3%（38/60）提升至 **93.3%（56/60）**（+30.0 pp，McNemar $p=1.2\times 10^{-4}$），Spatial 从 45.0% 升至 90.0%。
  - 完整 180 轮 Gemini + K1 评测（3 个初始状态）：总成功率 **77.2%（139/180）**（Object 86.7%，Spatial 81.7%，Goal 63.3%），松爪稳定性复检为 72.8%（131/180）。对照文献协议下其他方法：OpenVLA / $\pi_0$ / NORA 0.0%、MolmoAct 1.0%、X-VLA 3.3%、AtomVLA 6.7%、$\pi_{0.5}$ 12.8%、CaP-Agent0 23.2%、RATs 43.8%。
- **跨本体与跨仿真器零样本迁移（Section 4.4，表 2、表 S10–S15）：**
  - **RoboSuite：** 仅更换感知与控制适配器，4 项共享单臂任务平均成功率 Panda **90.0%**（72/80）、UR5e **88.8%**（71/80）、IIWA **86.2%**（69/80），其中 3 项方块任务三臂均 100%，螺母装配分别为 60.0%/55.0%/45.0%；Panda 全 7 项任务达 72.1%（101/140），双臂交接（Two-arm handover）仅 5.0%（1/20）成为主要瓶颈。
  - **RoboTwin（Aloha AgileX 双臂）：** 10 项任务宏平均 Easy **32.0%**（微平均 31/96 = 32.3%）、Hard **28.0%**（微平均 25/86 = 29.1%），仅下降 4.0 pp（对比文献中 RDT 从 43.1% 降至 15.9%，$\pi_0$ 从 64.1% 降至 23.0%）。
  - **RoboCasa（PandaOmron 移动操作）：** 24 项厨房任务 × 10 状态共 240 轮，总成功率 **22.1%（53/240）**，呈现明显方向不对称（如柜体取放至台面 50% vs 台面放入柜体 0%，关微波炉 50% vs 开微波炉 0%）。
- **真机实验（Section 4.5，表 3、Appendix D.4）：**
  - Piper 机械臂 + 两台外置 RGB-D 相机 + Gemini 3.7 Flash：Clean 基线（4 个工具）总计 1/9（11.1%），K1 总计 **8/9（88.9%）**（罐入碗 2/3、碗上盘 3/3、双物分拣 3/3）。原文特别说明：其中 1 次碗测试包含外部完成复核与夹爪单元引导，另 1 次操作员中断的罐测试单列记录，故该实验属于可行性验证而非全自主评测。
- **Agentic SFT 与混合环境续训（Section 4.6，表 4–5、表 S8–S9、表 S16）：**
  - 基于 107 条轨迹微调 5 个 epoch：Qwen K1 RUA 在 Split A（训练构型）/ B（新初始状态）/ C（未见任务条件）分别达 **51.2% / 44.2% / 13.9%**（5/36）；对比 OpenVLA-7B 为 20.9% / 30.2% / 0.0%，$\pi_{0.5}$ 为 7.0% / 7.0% / 0.0%，纯 RGB 文本动作 Qwen RUA 为 7.0% / 4.7% / 0.0%（35.2% 因非法动作格式终止），RGB-D Qwen VLA 为 0.0% / 0.0% / 0.0%。
  - Epoch-3 权重在 316 条四环境轨迹续训 3 epoch 后：LIBERO-PRO B 从 42.9% 升至 **52.4%**，C 从 11.1% 升至 **27.8%**，RoboSuite 从 28.3% 升至 **50.0%**，但 RoboCasa（8.3% → 4.2%）与 RoboTwin（2.6% → 2.5%）仍低于 10%。
- **消融与开销分析（Section 4.7，表 6、表 S17–S21）：**
  - 感知与完成反馈耦合（表 S17）：LIBERO-PRO 上二者皆关为 10.0%，仅开感知 30.0%，仅开反馈 13.3%，同时开启达 **70.0%**。
  - 工具族消融（表 6 / 表 S18）：LIBERO-PRO 移除跨帧追踪（即调用间锚点自动刷新）掉分最多（70.0% → 40.0%）；RoboTwin 移除区域工具或抓取候选掉分最多（30.0% → 13.3%）。
  - 记忆窗口（表 S20）：历史图像 $K=0$ 在 18 个样例上达 83.3%（28.3 次调用），而 $K=3$ 降至 61.1%（47.4 次调用）；文本窗口 $N=8$ 为 77.8%（33.0 次调用），缩短至 $N=1,2$ 虽保持 72.2% 但调用翻倍至 70.0–72.6 次。
  - 运行耗时（表 S21）：Astra+K1 平均 7.31 分钟/轮（45.9 次调用），Gemini+K1 平均 5.16 分钟/轮（34.9 次调用），模型 API 请求耗时占比达 77%–83%。

### 4. 局限性与知识映射

- **局限：** 依赖标定 RGB-D 与可见表面几何；工具只能暴露误差而不能自动修正策略（如盘子放置中出现残余 6.7 cm 误差并反复微调直至耗尽步数）；精细接触装配（螺母装配、插入容器）与双臂协调释放（handover 5.0%）仍是瓶颈；运行延迟受限于多轮 VLM API 往返；部分环境适配器与全部权重/数据集尚未随代码仓公开。
- **对 wiki 的映射：** [Robo-Harness K1](../../wiki/entities/paper-agentic-2609-29389.md)
