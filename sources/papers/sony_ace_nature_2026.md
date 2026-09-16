# Outplaying elite table tennis players with an autonomous robot（Nature 2026）

> 来源归档（ingest）

- **标题：** Outplaying elite table tennis players with an autonomous robot
- **项目名：** **Ace**（Sony AI 自主乒乓球机器人）
- **类型：** paper / real-world robotics / table-tennis / reinforcement-learning / event-based-vision
- **期刊：** Nature **652**, 886–891（2026-04-23）
- **DOI：** <https://doi.org/10.1038/s41586-026-10338-5>
- **Nature 页面：** <https://www.nature.com/articles/s41586-026-10338-5>
- **PDF：** <https://www.nature.com/articles/s41586-026-10338-5.pdf>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42020866/>（PMID: 42020866）
- **PMC 全文：** <https://pmc.ncbi.nlm.nih.gov/articles/PMC13102714/>（PMCID: PMC13102714）
- **项目页：** <https://sonyresearch.github.io/ace_public/> — 归档见 [`sources/sites/ace-sony-research-github-io.md`](../sites/ace-sony-research-github-io.md)
- **官方介绍：** <https://ai.sony/news/sony-ai-announces-breakthrough-research-in-real-world-artificial-intelligence-and-robotics>
- **Ace 品牌站：** <https://ace.ai.sony>
- **代码 / 补充材料：** <https://github.com/SonyResearch/ace_public> — 归档见 [`sources/repos/ace-public.md`](../repos/ace-public.md)
- **数据集：** <https://sonyresearch.github.io/ace_public/data/>（`match_data.csv`）
- **近似训练伪代码：** <https://sonyresearch.github.io/ace_public/pseudo_code/>
- **机构：** Sony AI（Zürich / Tokyo / New York）、Sony Advanced Visual Sensing、Sony Global Manufacturing & Operations、Sony Techno Create 等
- **入库日期：** 2026-09-16
- **一句话说明：** 首个在 **ITTF 正式规则、奥运尺寸场地** 下与 **精英/职业** 人类选手对打并取得 **多场胜利** 的自主乒乓球系统；**事件相机 spin 感知 + 非对称 SAC 技能库 + FAOC/MPC 轨迹安全层 + 定制 8-DOF 硬件** 闭环。

## 核心摘录（面向 wiki 编译）

### 1) 系统三件套

| 模块 | 要点 |
|------|------|
| **感知** | 9× APS（Sony IMX273）200 Hz 三角化，3 mm / 10.2 ms；3× **GCS**（IMX636 EVS + 电调远摄 + 振镜跟踪）估计 **球 spin**，CNN 低延迟 + CMax 高精度异步融合，~400–700 Hz |
| **控制（对打）** | 仿真训 **SAC** 非对称 actor–critic；**技能库**（落点/旋转类型）+ 赛中采样；31.25 Hz 查表 → **FAOC** 映射 32 ms 段轨迹 → **MPC reset**；碰撞预测则回退安全 reset |
| **控制（发球）** | 人类 demo 抛球 + 仿真 **遗传算法** 离线寻优拍面姿态/速度 + 专家评估 **发球库**（15 种对精英、13 种对职业） |
| **硬件** | 定制 **8-DOF**（2 平移 + 6 旋转）；Scalmalloy 拓扑优化连杆；1 kHz 同步；末端 Butterfly 胶皮 + 发球杯 |

### 2) 与既往乒乓球机器人差异

- **Spin 显式建模与测量**：Magnus 力、球–桌/球–拍接触模型在仿真中定制；spin 估计误差 ~24.8 rad/s。
- **非对称 actor–critic**：critic 用仿真真值球态，actor 用 **N 步噪声传感历史** → sim-to-real。
- **抽象动作空间 + 凸优化**：策略输出经 FAOC 映射为 **32 ms 终端约束**，再求 **无碰撞** 连续轨迹（1 kHz），而非直接关节 setpoint。
- **完整竞赛条件**：无改规则/缩小场地/禁发球等简化；JTTA 持证裁判；Nittaku 官方用球。

### 3) 2025-04 评测（论文 Fig. 3）

| 对手 | 赛制 | 结果 |
|------|------|------|
| 5 名 **精英**（≥10 年训练，平均每周 ~20 h） | BO3 | **Ace 赢 3/5 场**，13 局中 **7 胜** |
| 2 名 **职业**（T.League：安藤美奈み / 曾根翔） | BO5 | Ace **0/2 场**，7 局中 **1 胜** |

- **回球能力：** 对 ≤14 m/s 来球回球率与人类相当或更好；spin ≤450 rad/s 回球率 **>75%**；Ace 最高产出 **16.4 m/s / 600 rad/s**，可回对手最高 **19.6 m/s / 867 rad/s**。
- **得分模式：** 人类「制胜球」线/角速度分布高于「仅回球」；Ace 的 Returned/Won 分布相近 → **稳定回球 + 战术多样性** 而非单纯更快。
- **Ace（发球直接得分）：** 对精英 **16** 个 ace vs 人类合计 8；对职业 4 vs 7。
- **反应：** 触网后 **49 ms** 轨迹分叉并成功回球（Fig. 4）；平均回合 **5.0±3.0** 拍，长于典型人类 **3.9±2.0**。

### 4) 开源核查（项目页 / GitHub，截至 2026-09-16）

| 组件 | 状态 |
|------|------|
| 对打 **match 后事件球态 CSV** | ✅ [`match_data.csv`](https://github.com/SonyResearch/ace_public/blob/main/data/match_data.csv) |
| **近似 Python 伪代码**（SAC 训练环、rollout、FAOC、发球 GA、GCS） | ✅ `pseudo_code/`（TensorFlow 风格，非可运行完整栈） |
| 补充 **比赛 / GCS / 发球视频** | ✅ GitHub Pages |
| 训练权重、真机部署、感知栈、机器人接口 | ❌ 未发布 |
| 定制 **8-DOF 硬件** CAD / 控制固件 | ❌ 未发布 |

→ wiki 归类：**部分开源**（数据 + 教学伪代码；不可复现完整 Ace 系统）。

## 对 wiki 的映射

- 新建实体页：[`wiki/entities/paper-sony-ai-ace-table-tennis.md`](../../wiki/entities/paper-sony-ai-ace-table-tennis.md)
- 交叉更新：
  - [`wiki/methods/table-tennis-strategy-skill-learning.md`](../../wiki/methods/table-tennis-strategy-skill-learning.md) — 仿真分层乒乓球 vs **真机竞技** 对照轴
  - [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md) — 非对称 SAC + 技能库采样
  - [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md) — 噪声传感历史 → 真机零样本迁移案例

## 参考来源（原始）

- 论文：<https://doi.org/10.1038/s41586-026-10338-5>
- 项目 / 补充材料：<https://sonyresearch.github.io/ace_public/>
- GitHub：<https://github.com/SonyResearch/ace_public>
- Sony AI 新闻：<https://ai.sony/news/sony-ai-announces-breakthrough-research-in-real-world-artificial-intelligence-and-robotics>
