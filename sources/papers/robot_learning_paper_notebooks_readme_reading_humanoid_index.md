# Robot Learning Paper Notebooks 上游 README 通读

> 来源归档（原文通读 · 支撑 `wiki/overview/humanoid-paper-notebooks-index.md`）

- **标题：** 每日论文阅读计划（Robot_Learning_Paper_Notebooks 仓库 README）
- **作者：** ImChong
- **类型：** repo / awesome-list / paper-notebook
- **原文：** <https://github.com/ImChong/Robot_Learning_Paper_Notebooks>（2026-11 通过 raw.githubusercontent.com 获取 `main/README.md` 全文）
- **一句话说明：** 笔记站上游仓库的自述：规则、学习路线图（①–⑨ + 工程层）、基础 RL 论文与 MimicKit 官方源码对照表、高影响力精选清单、笔记结构与项目目录。
- **核对范围：** 通读 README 全文（徽标、来源、在线演示、规则、路线图、MimicKit 对照表、高影响力精选、笔记说明、项目结构）。未逐篇打开其中链接的 papers/ 笔记、PROGRESS.md、progress.json，也未核对各论文原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** 人形机器人学习方向的每日论文精读笔记，逐篇部署为 Jekyll 网页；论文清单来源为 YanjieZe/awesome-humanoid-robot-learning，待读清单在 `papers/PROGRESS.md`。
- **徽标口径（README 当日）：** Papers 1423、Notes 418；与本库快照的 518 条分类清单、289 条 HTML 笔记口径不同，引用时需标明口径与日期。
- **规则：** 每天 7:00 推送当日论文；回复「理解了」才进入下一篇；对话记录保存进对应 MD 笔记。
- **学习路线图：** ① 基础 RL（PPO→AWR，Transformer）→ 人体动作数据（AMASS/HumanML3D）→ 动作重定向（GMR 等，决定模仿质量上限）→ ② 精确模仿（DeepMimic→PHC）与风格学习（AMP→ADD/SMP）→ ③ 技能组合/扩散（ASE→CALM→PULSE，Diffusion Policy→BeyondMimic）→ ④ 全身控制（Expressive WBC、HOVER、H2O/OmniH2O、TWIST 等）→ 分出 ⑤ 操作、⑥ 移动操作、⑦ 世界模型、⑧ WAM → ⑨ 基础模型终点（VLA：OpenVLA/π₀/π₀.₅/GR00T N1；BFM）。另有横跨路线的状态估计、硬件与 Sim-to-Real 工程层（InEKF、Berkeley Humanoid、ToddlerBot、Domain Randomization、LCP、ASAP 等）。
- **MimicKit 对照表：** PPO/AWR/DeepMimic/AMP/ASE/ADD/SMP/LCP 与 MimicKit 本身标 ✅；PHC/CALM/PULSE/Diffusion Policy/BeyondMimic 有独立官方仓库、标 ❌；Domain Randomization 无官方代码。
- **高影响力精选：** 分全身控制核心、遥操作与模仿学习、行走经典、仿真到现实与基座模型、仿真平台与工具五组。
- **笔记结构：** 基本信息→一句话总结→英文缩写速查→问题背景→方法详解→具体实例→工程价值→（可选）MimicKit 源码对照→面试 Q&A→讨论记录，附录按需。
- **仓库结构：** `papers/` 按 14 个方向目录，另含 PROGRESS.md 与 todos/；`scripts/` 为预处理/部署脚本；Jekyll 配置 baseurl 为 `/Robot_Learning_Paper_Notebooks`。

## 对 wiki 的映射

- [`wiki/overview/humanoid-paper-notebooks-index.md`](../../wiki/overview/humanoid-paper-notebooks-index.md)
