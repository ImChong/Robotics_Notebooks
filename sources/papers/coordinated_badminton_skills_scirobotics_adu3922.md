# Learning coordinated badminton skills for legged manipulators（Science Robotics 2025）

> 来源归档（ingest）

- **标题：** Learning coordinated badminton skills for legged manipulators
- **类型：** paper / loco-manipulation / reinforcement-learning / visuomotor / sports-robotics / sim2real
- **期刊：** Science Robotics
- **DOI：** <https://doi.org/10.1126/scirobotics.adu3922>
- **论文页：** <https://www.science.org/doi/10.1126/scirobotics.adu3922>
- **项目页：** <https://articuno144.github.io/Learning-agile-badminton-skills/> — 归档 [`sources/sites/coordinated_badminton_eth_articuno.md`](../sites/coordinated_badminton_eth_articuno.md)
- **ETH 新闻：** <https://ethz.ch/en/news-and-events/eth-news/news/2025/08/playing-badminton-against-a-robot.html>
- **ETH Research Collection：** <https://www.research-collection.ethz.ch/items/ab676cdd-86d0-48b0-b255-303cb98f2485>
- **视频：** <https://youtu.be/zYuxOVQXVt8>
- **作者：** Yuntao Ma, Andrei Cramariuc, Farbod Farshidian, Marco Hutter
- **机构：** 苏黎世联邦理工学院（ETH Zurich）Robotic Systems Lab；Robotics and AI Institute（RAI）；Intel Labs；Max Planck ETH CLS；NCCR Robotics 等（见项目页致谢）
- **发表：** 2025-05-28（Crossref / 项目页）
- **入库日期：** 2026-09-26
- **一句话说明：** **统一 RL 全身 visuomotor 策略** 让 **ANYmal-D + DynaArm** 仅凭 **机载感知** 与人对打羽毛球；**感知噪声模型** 对齐 sim–real 并学出主动感知；含 **羽毛球预测 + 约束 RL**；真机多环境与人对练验证。

## 开源状态（步骤 2.5）

| 入口 | 2026-09-26 核查 |
|------|-----------------|
| 项目页 articuno144.github.io | Paper / Video / 文字说明；**无 GitHub / 权重** |
| ETH 新闻 | 报道与视频；**无代码链** |
| Research Collection | 机构元数据与全文入口；**无训练代码仓** |
| GitHub 检索 | 无 `leggedrobotics` / 作者公开 **badminton 训练栈** |
| **结论** | **未开源** — 复现依赖论文 + Demo 视频；工程集成须自研或等官方发布 |

## 核心摘录（面向 wiki 编译）

### 1) 任务与统一策略（摘要 + 项目页）

- **任务：** 动态环境中协调 **下肢 locomotion、上肢挥拍、视觉感知**；羽毛球需 **追踪 + 击球**。
- **方法：** 单一 **RL 全身 visuomotor 策略**（全 DoF）；**非对称 actor–critic** — 部署策略仅见机载可观测量，critic 用仿真特权信息估价值。
- **感知：** **Perception noise model** 用真机相机数据标定噪声，使 sim/deploy **感知误差水平一致**；鼓励 **主动感知**（如 pitch 保球可见再下压挥拍），无需硬编码 FOV 约束。
- **控制增强：** **Shuttlecock prediction model** + **constrained RL** 提升部署鲁棒性。

### 2) 平台与真机表现（项目页）

- **硬件：** **ANYmal-D** + **DynaArm** 操作臂；**仅机载感知** 自主与人对打。
- **定量/现象：** 单回合最多 **10 连拍**；挥拍速度最高 **12.06 m/s**；距/时约束下 **步态自适应**（近场少动、远场 gallop）；击球后 **回中场**；户外有风环境可运行。

### 3) 与人形羽毛球文献区分

| 维度 | 本文（ETH ANYmal） | [人形 Multi-Stage RL](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md) | [LHBS 拟人羽毛球](../../wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md) |
|------|-------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| 形态 | 四足 + 臂 | 人形 Phybot C1 | 人形 + AMP/DAgger |
| 期刊 | Science Robotics | arXiv | arXiv |
| 感知 | 机载相机 + 噪声模型 + 主动感知 | MoCap/EKF 或短历史 | 仿真为主 + sim2real |
| 开源 | 未开源 | 宣称待发布 | 依 EngineAI/HKU 条目 |

## 对 wiki 的映射

- **升格实体页：** [`wiki/entities/paper-coordinated-badminton-skills-anymal.md`](../../wiki/entities/paper-coordinated-badminton-skills-anymal.md)
- **站点归档：** [`sources/sites/coordinated_badminton_eth_articuno.md`](../sites/coordinated_badminton_eth_articuno.md)
- **交叉：** [ANYmal](../../wiki/entities/anymal.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[人形羽毛球 Multi-Stage](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md)、[depth-humanoid-boxing Stage 5](../../roadmap/depth-humanoid-boxing.md)
