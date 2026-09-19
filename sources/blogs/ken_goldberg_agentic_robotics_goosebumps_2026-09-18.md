# Goosebumps: a Paradigm Shift is Occurring in Robotics（Ken Goldberg X Article）

> 来源归档（social / X Article + ICRA 2026 plenary 摘要）

- **标题：** Goosebumps: a Paradigm Shift is Occurring in Robotics
- **类型：** blog / commentary / plenary-summary
- **作者：** Ken Goldberg（@Ken_Goldberg，UC Berkeley 教授；Ambi Robotics / Jacobi Robotics 联合创始人）
- **原始链接：** <https://x.com/ken_goldberg/status/2100986412762087909>（X Article: <https://x.com/i/article/2100968281100451840>）
- **Plenary 视频：** <https://bit.ly/Agentic-Robotics-plenary-by-Ken-Goldberg>（ICRA 2026，2026-06-02，Vienna，*A Tale of Two Cultures: Can Agentic Coding Close the Gap?*，约 45 分钟）
- **发表日期：** 2026-09-18
- **入库日期：** 2026-09-19
- **抓取方式：** fxtwitter API 提取 X Article 全文 + plenary 转录摘要
- **一句话说明：** Goldberg 提出 **Agentic Robotics（AR）** 为 model-based 工程与 model-free VLA 之间的 **第三条路**：多 agent **离线** 编写/测试/迭代 **结构化机器人程序**（GaP 计算图），并与 **逆物理 Real2Sim**、产线 **数据雪崩** 闭环；主张 **不是 VLA vs World Model**，量产需 **二者 + 传统方法**，由 agentic coding 集成。

## 开源核查

本文为 **社交媒体长文 + 演讲策展**，非单一项目页。**GaP** 官方代码已开源（见 [graph-as-policy](../../sources/repos/graph_robots_graph_as_policy.md)）；文中 **Robot Sim Studio（RSS）+ Newton**、**Ambi 工业部署** 为组内/商业进展，**截至入库日无统一公开仓库**——wiki 只写已核实链接，不臆造 repo。

## 核心摘录（归纳，非全文）

### 1. Agentic Robotics（AR）定义

- **范式位移：** 不是换一种 policy 架构，而是 **机器人智能在哪里被开发**——从「海量真机 demo」或「工程师逐任务手工编程」转向 **多 agent 离线写、测、诊断、迭代结构化程序**。
- **第三路径：** 介于 **model-based（GOFI：快、可解释、可靠、但每任务人工重）** 与 **model-free VLA（潜在泛化、需海量数据、工业可靠性不足）** 之间。
- **关键性质：** AR **不依赖 demonstration data**（脚注引用 Science Robotics 2025「100,000 Year Robot Data Gap」）；用 frontier LLM/VLM **组合模块化 skills**（可含 VLA 子模块）为 **可解释控制系统**；**运行期** 导出 **轻量可执行代码**，不必在 inner loop 跑 frontier 模型。

### 2. 场内的「两种文化」与整合主张

- **Model-based vs Model-free：** 传统 handbook/ROS/TAMP 路线 vs 2012 起 deep learning → transformer → 2023 **VLA** 端到端。
- **Tweet / plenary 核心句：** **不是 VLA vs World Model**；量产机器人需要 **VLA + World Model + model-based 方法**，全部由 **agentic coding** 集成。
- **Specialist vs Generalist：** 通才机器人融资巨大但 **paid useful work ≈ rounding error**（引 York Gang 博文）；**产线真实部署更接近 specialist / variational automation**。
- **数据缺口：** 相对 LLM 训练 token，机器人数据约 **5 个数量级** 差距；**数据 alone 不够**（Waymo vs Tesla 反例：更多 miles ≠ 更好，因 **模块化 GOFI** 仍关键）。

### 3. GaP（Graph-as-Policy）与 CaP-X

- **2026-05 GaP：** 图结构 AR harness，多 agent 分节点写 **ROS2 兼容** 计算图，仿真排练自学习，导出 edge 执行（详见 [GaP 论文](../../sources/papers/gap_arxiv_2607_05369.md)）。
- **CaP-X（ICML 2026）：** 单 agent 写代码 baseline ~32%，RL 自修正可提升；GaP 在 VA benchmark 上 **显著高于 VLA / TipTop / CaP**；**π₀.₅ + GaP staging** 可 **2–3×** 裸 VLA 成功率。
- **Graph as Policy vs Code as Policy：** 图 = 类型检查 + 多 agent 分工 + 与 ROS/OpenCV 等 **组合原语** 对齐；消融：无图或单 agent → **0%**。

### 4. 数据雪崩（Data Avalanche）与产线飞轮

- **Ambi Robotics：** 累计 **1 亿+** 包裹分拣；保存每次 pick 成功/失败记录；对 **deformable bags** 用 **22 人年等价** 产线数据训练生成模型，优于 DexNet 刚性物体先验。
- **Flywheel → Avalanche：** Leslie Kaelbling / Pok 点出产线数据不是稳定飞轮而是 **持续增长雪崩**——部署 → 采集 → 改进 → 多卖 → 再采集。

### 5. Real2Sim2Real 与「Goosebumps」时刻

- **逆物理瓶颈：** 自学习若全靠真机则慢、贵、易损；仿真需 **inverse physics**（几何/摩擦/质量/阻尼/可变形体）——传统需大量人工，区别于 Real2Sim / SysID 但相关。
- **Robot Sim Studio（RSS）：** Kaiyuan Chen 开发的 **Newton + Viser** 交互调参前端，原设想服务 **人类工程师**。
- **2026-09-03 GPT-6 Astra：** 给 RSS + 实验室海绵擦金属条短视频，**<1 小时** 建 Newton 模型复现可变形接触；再交 GaP harness 自学习图 → **真机可跑**。
- **失败即证据：** 一次真机推倒金属条 → 质量不可从视觉推断 → agent 可用失败视频 **更新仿真/增鲁棒** 再部署——AR 闭环特征。
- **2026-09-17 Jeff Mahler（Ambi 联创）：** 宣布 GaP AR harness 已解决 **真实工业分拣问题** 并部署全美机器人。

### 6. 结论与开放问题

- **范式：** 瓶颈可能从「够多 robot data / 够多工程师」转向 **为 robot-engineering agent 指定目标、约束、接口与 V&V 准则**。
- **Goldberg 态度转变：** 长期对 near-term 机器人落地偏 cautious，现认为 **real robots timeline 显著提前**。
- **未解决：** 物理模型仍 imperfect；主动/课程自学习；仿真+真机鲁棒与速度；安全关键 V&V；coding agent 仍慢（仅适合 **离线**）；能否 scale 到开放通才/人形 **未定**（脚注 6）。

## 对 wiki 的映射

- **主实体页：** [Ken Goldberg：Agentic Robotics 范式位移](../../wiki/entities/ken-goldberg-agentic-robotics-goosebumps.md)
- 交叉：[GaP 论文实体](../../wiki/entities/paper-gap-graph-as-policy.md)、[变体自动化（VA）](../../wiki/concepts/variational-automation.md)、[VLA](../../wiki/methods/vla.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[Data Flywheel](../../wiki/concepts/data-flywheel.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[ASPIRE](../../wiki/methods/aspire.md)、[GPT-6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)
