# Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation

> 来源归档（ingest）

- **标题：** Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation
- **缩写：** **Uni-LaViRA** / Language–Vision–Robot Actions Translation
- **类型：** paper / vln / objectnav / eqa / aerial-vln / zero-shot / agentic / mllm / multi-embodiment
- **arXiv：** <https://arxiv.org/abs/2605.27582>（PDF: <https://arxiv.org/pdf/2605.27582.pdf>）
- **HTML：** <https://arxiv.org/html/2605.27582>
- **项目页：** <https://xetroubadour.github.io/Uni-LaViRA/> — 归档见 [`sources/sites/xetroubadour-uni-lavira-github-io.md`](../sites/xetroubadour-uni-lavira-github-io.md)
- **代码：** <https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code> — 归档见 [`sources/repos/uni-lavira-code.md`](../repos/uni-lavira-code.md)
- **机构：** 南京大学（NJU）；中国科学院自动化研究所（CASIA）；北京航空航天大学（BUAA）；宝马南京信息技术（BMW Nanjing Information Technology）；罗切斯特大学（University of Rochester）
- **作者：** Hongyu Ding\*、Sizhuo Zhang\*、Ziming Xu\*、Jinwen Guo、Hongxiu Liu、Xingzhi Cheng、Zixuan Chen、Haifei Qi、Duo Wang、Hao Xu、Jieqi Shi†、Yifan Zhang†、Jing Huo†、Jian Cheng、Yang Gao、Jiebo Luo（\*共同一作；†通讯）
- **前序工作：** LaViRA（arXiv:2510.19655，项目页 <https://robo-lavira.github.io/lavira-zs-vln/>）— 单任务 VLN-CE 验证三层翻译
- **硬件（真机）：** Agilex Cobot Magic（轮式双臂）、Unitree G1（人形）、Unitree Go1（四足）、自研四旋翼 UAV
- **状态：** arXiv 预印本（约 2026-05）
- **入库日期：** 2026-07-22
- **一句话说明：** 将具身导航决策结构化为 **Language → Vision → Robot Action** 三层翻译；用预训练 MLLM 做零训练统一 agent，覆盖四任务族与四异质本体，并引入 TDM / SCB 闭环机制。

## 摘录 1：问题与立场

- **动机：** 近年导航通才路线主要靠 **扩大机器人轨迹与 VLA 训练规模**；本文主张对导航而言，**结构性分解** 也可获得跨任务 / 跨本体一般性。
- **核心观察：** VLN-CE / ObjectNav / EQA / Aerial-VLN 共享同一决策骨架——理解指令 → 在当前观测接地 → 发出空间动作。语言层语义方向与视觉层像素目标，均落在预训练 MLLM 的 **natural output manifold** 内，因而可 **推理** 而非从机器人数据学习。
- **物理边界：** 接触丰富的灵巧操作动作（力矩 / 接触力 / 阻抗）常在 MLLM 流形外，仍需端到端 VLA；主流导航多为 **contact-free 空间推理**，故可完整继承预训练 MLLM 泛化。
- **相对 LaViRA：** 会议版仅 VLN-CE + 两本体 + 朴素回退；本版扩至 **4 任务 × 4 本体**，并新增 **TDM**（长指令工作记忆）与 **SCB**（错误条件化重规划）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-uni-lavira.md`](../../wiki/entities/paper-uni-lavira.md)；交叉 [VLN](../../wiki/tasks/vision-language-navigation.md)、[VLA](../../wiki/methods/vla.md)、[四范式复现](../../wiki/overview/vln-open-source-repro-paradigms.md)。

## 摘录 2：三层架构与统一接口

每步输入：任务规格 \(\mathcal{T}\)（路线 / 物体类 / 问题 / 飞行指令）、egocentric RGB-D \(\mathcal{O}_t\)、位姿、结构化历史 \(\mathcal{H}_t\)；输出本体原生动作 \(\mathcal{A}_t\)。

1. **Language Action \(\phi_{\mathrm{lang}}\)**（实验：Gemini-3.1-Pro）：四向全景 + 历史 → 工具调用 `turn(dir)` / `backtrack(wp)` / `go_stair` / `double_check(stop)`，附 JSON 推理与 TDM 更新。
2. **Vision Action \(\phi_{\mathrm{vis}}\)**（实验：Qwen3.5-27B）：在所选视角上 `select(bbox/point)` + `target_desc`；**不依赖** 预训练 waypoint predictor。
3. **Robot Action \(\pi_{\mathrm{robot}}\)**：bbox 像素 + 深度反投影 → 世界坐标 → 短视界规划（地面 FMM 2D 占据；UAV 3D 体素 + 可见性图）→ 本体原生控制器。**仅此层与本体绑定。**

**对 wiki 的映射：** 实体页「流程总览 / 核心原理」；与训练式导航 VLA（Uni-NaVid 等）对照。

## 摘录 3：TDM 与 SCB

- **TODO List Memory（TDM）：** 每项 `(content, status∈{pending,completed}, result)`；episode 初一次初始化，之后每步在同一 LA 调用中先输出 update/rewrite/add/remove，再选动作；把未完成子目标 **写回最近注意力窗口**（针对 RxR ≈120 词指令与多段 UAV 航线）。纯 prompt，无新参数。
- **Second Chance Backtrack（SCB）：** `backtrack(wp_k)` 沿最新占据图回到先验航点；再以 **失败方向 + 失败子轨迹图像序列** 条件化重规划，而非丢弃失败证据后盲重试（对比 SmartWay / LaViRA 的一步撤销）。

**对 wiki 的映射：** 实体页 agent-loop；工程实践「长时程记忆 / 纠错」。

## 摘录 4：实验、真机与开源

- **协议：** 各 benchmark **分层抽样 100-episode** val-unseen（OpenUAV 为 UM）；三随机种子；相对全量 split 做基线对齐（多数指标 ±2 内）。
- **零训练主结果（均值）：** R2R SR **60.7%**、RxR SR **51.3%**、HM3D-v2 SR **77.7%**、HM3D-OVON SR **60.0%**、MP3D-EQA ACC **54.7%**、OpenUAV SR **40.0%**——零样本块内强，并可对标部分需百万级轨迹的导航基础模型。
- **真机：** 同一 LA/VA 核心，仅换低层控制器，部署到 Cobot Magic / Go1 / G1 / 自研 UAV。
- **开源（项目页核查，截至 2026-07-22）：** **已开源** — 仿真（Habitat + AirSim）评测与真机部署代码齐全；License **CC BY-NC-SA 4.0**；依赖外部 MLLM API（LA/VA keys）与场景数据集许可。

**对 wiki 的映射：** sites / repos 归档；实体页开源状态与源码运行时序图。

## 当前提炼状态

- [x] arXiv / 项目页 / GitHub 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-uni-lavira.md` 新建
- [x] 开源边界写入 sites / repos / wiki（可运行仿真评测 + 真机入口）
