# Humanoid Motion Intelligence 根 README 通读

> 来源归档（原文通读 · 支撑 `wiki/entities/humanoid-motion-intelligence.md`）

- **标题：** 人形机器人运动智能知识库（RealXiaoze/humanoid-motion-intelligence 根 README.md）
- **作者 / 维护方：** 具身智能研究室（GitHub `RealXiaoze`）
- **类型：** repo / curated knowledge base README
- **原文：** <https://raw.githubusercontent.com/RealXiaoze/humanoid-motion-intelligence/main/README.md>（2026-11 经代理获取 main 分支全文，120 行）
- **一句话说明：** 策展知识库的总入口：资料导航表 + 五大研究方向（各含 6 或 4 条技术路线、核心思路与代表项目）+ 使用方式（可交给 Agent 检索）+ 许可与交流方式。
- **核对范围：** 通读根 README 全文（资料导航表、五大方向全部技术路线表、「关于知识库」、克隆与许可段）。README 内链接的子页面（技术与研究/、论文逐篇解读/、数据集/、公司主表等）未逐页阅读，仅按 README 描述引用。注意：2026-11 的 README 已改为「五大方向」（动作数据与跨本体迁移 / 运动控制与全身行为 / 物体交互与全身操作 / 世界模型与通用动作策略 / 任务规划与长程自主执行），论文编号已出现 P5xx，与 wiki 页 2026-09-10 快照的「六条路线 · 191 篇」口径不同。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **资料导航：** 10 个入口——论文与技术报告解读、技术与研究、具身智能数据集、公司与产品主表、公司开源项目、仿真/训练/开发工具、工程与实机部署、书籍与课程、求职与职业发展、全部资料目录。
- **方向 1 动作数据与跨本体迁移（6 路线）：** 人体/交互运动恢复、条件动作生成、运动学与学习式重定向、身体表面与接触迁移、可执行性与时间优化、数据生成与扩增。代表：GVHMR、WHAM、GMR、UMR、HOI-Retarget、DynaRetarget、HumanoidMimicGen。
- **方向 2 运动控制与全身行为（6 路线）：** 模型规划与 WBC、自主运动与自然步态（AMP/GMP）、参考跟踪与遥操作（H2O、OmniH2O、BeyondMimic）、技能表示与行为基座（ASE、PULSE、SONIC、BFM-Zero）、感知与场景适配、抗扰恢复与保护。
- **方向 3 物体交互与全身操作（6 路线）：** 交互模仿、接触规划与执行、任务强化学习、分层技能、精确柔顺与触觉、动态适配与状态预测。
- **方向 4 世界模型与通用动作策略（5 路线）：** 状态潜空间世界模型（PlaNet/Dreamer）、动作条件视觉未来、视觉模仿与动作生成（Diffusion Policy、ACT）、VLA（RT-2、π0、GR00T N1、OpenVLA）、WAM。
- **方向 5 任务规划与长程自主执行（4 路线）：** 任务分解与技能调用、空间理解与主动观察、执行核验与重规划、记忆与技能积累。
- **横向主题：** 感知与状态估计、适应与持续改进、Sim2Real、安全与评测贯穿多方向，集中在「工程与实机部署」。
- **使用方式：** 可 `git clone` 后交给能读文件的 Agent，结合本体型号与研究问题检索；引用约定见 AGENTS.md。
- **许可：** 见仓库 LICENSE.md；欢迎经 GitHub Issues 反馈错误并附来源链接。

## 对 wiki 的映射

- [`wiki/entities/humanoid-motion-intelligence.md`](../../wiki/entities/humanoid-motion-intelligence.md)

## 参考来源

- 上游仓库：<https://github.com/RealXiaoze/humanoid-motion-intelligence>
