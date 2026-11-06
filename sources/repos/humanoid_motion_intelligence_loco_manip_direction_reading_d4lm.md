# humanoid-motion-intelligence：README 与「物体交互与全身操作」方向页通读

> 来源归档（原文通读 · 支撑 `wiki/overview/humanoid-motion-intelligence-day4-loco-manipulation.md`）

- **标题：** 人形机器人运动智能知识库（README）；技术与研究 / 研究方向_物体交互与全身操作.md
- **作者：** RealXiaoze
- **类型：** repo / curated-list / humanoid / loco-manipulation
- **原文：** <https://github.com/RealXiaoze/humanoid-motion-intelligence>（2026-11 经 raw.githubusercontent.com 获取 main 分支 `README.md` 与 `技术与研究/研究方向_物体交互与全身操作.md` 全文）
- **一句话说明：** Day 4 导读所依据的上游项目；README 给出五大研究方向索引，「物体交互与全身操作」方向页把移动操作相关工作分为 6 条技术路线并列出论文与项目目录。
- **核对范围：** 通读上述两份文件全文（README 约 30 KB，方向页约 55 KB，含 6 条路线表、各路线论文/项目表与按年份的相关论文目录）。未阅读 Day 4 微信公众号原文（经代理抓取返回错误页，无法获取）、各单篇论文解读（论文逐篇解读/P*.md）及各论文原文，页面中单篇工作的细节仅依赖本仓库独立详情页与该方向页的一句话摘要。
- **通读：** 已完成

## 核心摘录（均出自上游文件）

- **方向定义：** 该方向研究如何通过接触改变物体状态，同时协调身体运动、受力与平衡；击球、运球、搬运、开门、灵巧操作都涉及接触位置、时机、作用力与物体运动，移动操作还需协调腿、腰、手；部分方法来自物理仿真人形角色研究。
- **六条技术路线：** 交互模仿（HumanX、HDMI、PhysHOI、SkillMimic 系列）；接触规划与执行（OmniContact、FARO、MotionDisco、HITTER）；任务强化学习（DoorMan、MPC-RL、PACE、SMASH）；分层技能（SkillBlender、CoorDex、LATENT）；精确柔顺与触觉（GentleHumanoid、WT-UMI、CEER2、TouchGuide、ResGAC）；动态适配与状态预测（HAIC、SplitAdapter、DexNDM）。
- **柔顺路线：** 阻抗或导纳控制调节接触中的刚度与让步程度，视觉、力、触觉反馈进一步修正动作；CEER2 以固定全身跟踪策略为底座，分层残差策略调节手端三轴刚度与根部抗力/阻尼，Isaac Sim 训练后迁移 Unitree G1。
- **分层技能：** 高层决定如何组合已有技能，低层稳定执行；SkillBlender 用目标条件技能原语加逐关节混合器，SkillBench 提供 H1、G1、H1-2 配置。
- **任务 RL：** DoorMan 以分阶段 reset 训练特权开门教师，经 DAgger 蒸馏为 RGB 学生，再用 GRPO 改善部分可观测下的闭环行为，并随机化门体、材质、光照、相机。
- **动态适配：** 负载、摩擦、执行器变化使同一命令产生不同运动；HAIC 递推估计物体位姿/速度，SplitAdapter 分别编码负载与本体动力学。
- **README 结构：** 五大方向（动作数据与跨本体迁移、运动控制与全身行为、物体交互与全身操作、世界模型与通用动作策略、任务规划与长程自主执行）；感知与状态估计、Sim2Real、安全与评测贯穿各方向，集中在「工程与实机部署」。

## 对 wiki 的映射

- [`wiki/overview/humanoid-motion-intelligence-day4-loco-manipulation.md`](../../wiki/overview/humanoid-motion-intelligence-day4-loco-manipulation.md)
