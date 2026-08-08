# 具身智能高频面试题库（GitHub Pages）

> 来源归档（site / project-page）

- **标题：** 具身智能高频面试题库 · Embodied-AI Interview QA Bank
- **类型：** site / interview-qa-bank
- **官方入口：** <https://winstonjq.github.io/embodied-interview-qa/>
- **代码：** <https://github.com/WinstonJQ/embodied-interview-qa>
- **License：** MIT
- **维护者：** WinstonJQ
- **入库日期：** 2026-08-08
- **一句话说明：** 中文具身智能秋招高频面试题库站点：八卷折叠式问答（通识 / RL / VLA·IL / 世界模型·Sim2Real / 工程落地 / 腿足控制 / 感知导航 / LeetCode·系统设计），答案短、按频次与难度分层。

## 页面公开信息（检索自 2026-08-08）

| 资源 | URL |
|------|-----|
| 主册 | <https://winstonjq.github.io/embodied-interview-qa/> |
| 卷一 通识基础 | <https://winstonjq.github.io/embodied-interview-qa/interviews/01_basics.html> |
| 卷二 RL 算法 | <https://winstonjq.github.io/embodied-interview-qa/interviews/02_rl_algo.html> |
| 卷三 VLA / 模仿学习 | <https://winstonjq.github.io/embodied-interview-qa/interviews/03_vla_il.html> |
| 卷四 世界模型 / Sim2Real | <https://winstonjq.github.io/embodied-interview-qa/interviews/04_world_sim.html> |
| 卷五 工程落地 | <https://winstonjq.github.io/embodied-interview-qa/interviews/05_engineering.html> |
| 卷六 腿足控制 / 遥操作 | <https://winstonjq.github.io/embodied-interview-qa/interviews/06_legged_control.html> |
| 卷七 3D 感知 / SLAM / VLN | <https://winstonjq.github.io/embodied-interview-qa/interviews/07_perception_nav.html> |
| 卷八 LeetCode + 系统设计 | <https://winstonjq.github.io/embodied-interview-qa/interviews/08_coding_systemdesign.html> |
| 源码仓 | <https://github.com/WinstonJQ/embodied-interview-qa> |

## 开源核查（步骤 2.5）

| 维度 | 状态 |
|------|------|
| **内容开放** | GitHub Pages **公开可读**；题库 Markdown + HTML 均在仓内 `docs/interviews/` |
| **代码仓** | **已开源** — <https://github.com/WinstonJQ/embodied-interview-qa>（MIT；入库日约 134★） |
| **训练权重 / 数据集** | 不适用（面试题库，非算法模型仓） |
| **可运行入口** | 静态站点；无训练/推理脚本；贡献流程见 README（Issue / PR 追加 `<details class="qa">`） |

## 站点摘录（2026-08-08）

### 定位

- 面向 **VLA / IL / RL / 世界模型 / 工程落地** 等具身算法岗的中文高频题库。
- 题源：牛客、知乎、小红书、一亩三分地、GitHub 公开面经等；**同义题合并后频次 ≥3 入主表**；未满三源用「补充」标签，不伪造频次。
- 答案目标 **≤350 字** +「易错」一句；默认 HTML5 `<details>` 折叠，手机可刷。
- 难度：L1 必会 / L2 进阶 / L3 顶级 lab；另散布 §H 手撕代码题。

### 八卷规模（站点/README 宣称）

| 卷 | 主题 | 题数（宣称） |
|----|------|-------------|
| 01 | 通识基础（含手撕） | 55 |
| 02 | RL 算法（含手撕） | 50 |
| 03 | VLA / 模仿学习（含手撕） | 77 |
| 04 | 世界模型 / Sim2Real | 31 |
| 05 | 工程落地（含手撕） | 47 |
| 06 | 腿足控制 / 全身控制 / 遥操作（含手撕） | 58 |
| 07 | 3D 感知 / SLAM / VLN / ObjectNav / Embodied VLM（含手撕） | 67 |
| 08 | LeetCode 高频 + 系统设计 | 40 |

README 称主表及补充共约 **425** 题；入库日对 Markdown `<summary>` 计数约 **438**（含手撕与补充，以仓内文件为准）。

### 质量与生成说明（项目自述）

- 答案经「执行者 ≠ 审查者」跨模型二次审查（Claude + Codex/GPT）后发布。
- 题库本身用 multi-agent vibe coding 流水线维护；**不是**论文复现代码。

## 为什么值得保留

- **面试前补盲区入口**：与本库方法/概念深读互补——题库给短答案与频次信号，wiki 给机制与开源核查。
- **覆盖本库主线**：VLA、RL、Sim2Real、腿足 WBC、VLN、工程部署与本库 `wiki/methods` / `wiki/concepts` / roadmap 高度对齐。
- **MIT 可引用、可 fork**：内容与渲染脚本均开放，便于后续对照更新。

## 对 wiki 的映射

- 主升格：[`wiki/entities/embodied-interview-qa.md`](../../wiki/entities/embodied-interview-qa.md)
- 仓归档：[`sources/repos/embodied-interview-qa.md`](../repos/embodied-interview-qa.md)
- 交叉：[`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)、[`wiki/methods/imitation-learning.md`](../../wiki/methods/imitation-learning.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/concepts/whole-body-control.md`](../../wiki/concepts/whole-body-control.md)、[`wiki/entities/lumina-embodied.md`](../../wiki/entities/lumina-embodied.md)、[`wiki/entities/learn-robotics-qqfly-guide.md`](../../wiki/entities/learn-robotics-qqfly-guide.md)
