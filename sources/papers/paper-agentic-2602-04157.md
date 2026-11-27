# A Modern System Recipe for Situated Embodied Human-Robot Conversation with Real-Time Multimodal LLMs and Tool-Calling

- **作者：** Dong Won Lee, Sarah Gillet, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park
- **机构：** Massachusetts Institute of Technology (MIT); Carnegie Mellon University (CMU)
- **版本 / 载体：** arXiv:2602.04157v1 [cs.RO]（2026-02-04 提交，9 页，7 图）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2602.04157) / [arXiv Abs](https://arxiv.org/abs/2602.04157)
- **项目页 / 代码：** 未确认开源代码（arXiv:2602.04157v1 正文及附录未列出 GitHub 仓库、模型权重或独立项目页 URL；正文 §IV 与 §VI-D 仅提及补充材料视频）
- **核对范围：** arXiv:2602.04157v1 全文（Abstract、§I–§VII、Algorithm 1–4、Table I–IV、Fig. 1–3 文字说明及 Appendix -A 几何绑定层推导）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与系统配方：** 具身情境对话要求机器人在严格延迟约束下交替推进实时对话与主动感知（决定看什么、何时看、说什么）。论文提出由「实时多模态大模型（OpenAI Realtime / Gemini Live）+ 5 个注意力与主动感知工具原语 + SE(3) 几何绑定层」构成的极简系统配方，并将对话时序 mechanics（VAD、轮次切换、打断/barge-in、流式 ASR）交给实时模型运行时处理。
- **五类工具原语（§III-B，Algorithm 1–4）：**
  1. `use_vision(ws, q)`：按需截取当前相机帧 $I$ 连同查询 $q$ 通过 WebSocket 发送给实时模型，避免持续流式传帧带来的图像 token 累积成本。
  2. `look_at_person(enabled)`（文中部分处简写为 `look_at_me`）：无其他工具调用时的默认行为；本地运行 YOLO11-pose（$49.3 \pm 3.50\text{ ms}$，$20.37\text{ Hz}$）提取鼻部等关键点，若遮挡或超出视野则回退到空间音频定位 $A$。
  3. `look_at_object`：本地运行 SAM3（$161.7 \pm 4.31\text{ ms}$，$6.18\text{ Hz}$）提取目标掩码质心或检测框中心，闭环追踪所指物体。
  4. `look_around(C, D)`：按预定义目标点列表 $C$ 扫描环境，等待运动到位后保存 $(I_i, T_i, \mathrm{fov}_i, \mathbf{p}_i)$ 至本地视点记忆库 $D$。
  5. `look_for(q, D)`：针对语言查询 $q$ 在 $|\mathcal{V}|$ 个并发线程中用 VLM 对 $D$ 中各视点打相关性分 $s_i \in [0, 1]$，选取 $i^* = \arg\max_i s_i$ 驱动机器人转向 $\mathbf{p}_{i^*}$，并将选中帧 $I_{i^*}$ 与 $q$ 送回实时模型。
- **几何绑定层（Appendix -A）：** 利用运动学/IMU 获得的刚体变换 ${}^{R}\mathbf{T}_{C}(t) \in SE(3)$，通过针孔模型将归一化图像坐标 $(u_n, v_n)$ 反投影为相机射线 $\mathbf{r}_C$，计算偏航/俯仰误差 $\Delta\mathrm{yaw} = \arctan2(r_x, r_z)$、$\Delta\mathrm{pitch} = -\arctan2(r_y, r_z)$，并沿机器人坐标系射线投影标称距离 $d$（无深度估计时取 $1\text{–}2\text{ m}$）得到 3D 注视目标 ${}^{R}\mathbf{g}(t)$。
- **实验与关键数值（§IV–§V）：**
  - 在 Jibo 社交机器人上测试 6 类家庭场景（Table I：Posture coach、Whiteboard、Lamp placement、Plant doctor、Outfit check、Pack / find objects，单段时长 01:39–02:21）与 4 种系统配置（OpenAI Realtime、w/o Object、w/o Person、Gemini Live），共 24 段视频；另在 Reachy Mini 上展示跨本体迁移能力（§VI-D）。
  - 轮次级工具决策（Table II）：OpenAI Realtime 宏平均准确率 $0.72 \pm 0.10$（P=$0.88 \pm 0.18$, R=$0.62 \pm 0.23$），Gemini Live 准确率 $0.77 \pm 0.07$（P=$0.84 \pm 0.16$, R=$0.60 \pm 0.22$）；`look_for()` 精确率均达 $1.00$（R 分别为 $0.70 / 0.60$），`use_vision()` 精确率 $1.00$ 但召回偏保守（R=$0.60 / 0.62$），`look_at_object()` 召回成为主要瓶颈（R=$0.19 / 0.17$）。双标注员在 4 段视频上的一致性 Cohen's $\kappa = 0.41$。
  - 主观交互质量（Table III，5 分制）：对话语义连贯性（Convo. Quality）各变体均接近上限（均值 $4.65 \pm 0.15$），而禁用物体工具使 Reference to Object 从 $4.83 \pm 0.33$ 降至 $3.47 \pm 0.96$，禁用人物追踪使 Attention to Person 从 $4.75 \pm 0.34$ 降至 $4.12 \pm 1.03$。
  - 延迟与成本（Table IV）：端到端延迟相近（OpenAI $689.98 \pm 389.08\text{ ms}$ vs. Gemini $739.10 \pm 593.45\text{ ms}$），单次交互 API 成本差异明显（OpenAI $\$0.047 \pm 0.013$ vs. Gemini $\$0.018 \pm 0.002$）。
- **核查边界：** 论文结果基于 6 个作者选定场景、24 段录制视频及 2 名知晓系统功能但对变体盲测的标注员；端到端原型存在偶发感知失败或工具误触发等工程瑕疵（§VI-E）。
- **对 wiki 的映射：** [A Modern System Recipe for Situated Embodied Human-Robot Conversation with Real-Time Multimodal LLMs and Tool-Calling](../../wiki/entities/paper-agentic-2602-04157.md)
