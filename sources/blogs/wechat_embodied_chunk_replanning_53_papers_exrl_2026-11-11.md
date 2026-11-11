# 机器人动作块太长，什么时候该重新规划？｜附代码与项目入口（具身智能小站 · 53 篇）

> 来源归档（blog / 微信公众号）

- **标题：** 机器人动作块太长，什么时候该重新规划？｜附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/jhHMptmxYuIHfDUjTsmdtw
- **发表日期：** 2026-11-11
- **入库日期：** 2026-11-11
- **抓取方式：** 用 curl 取微信文章页 HTML，抽取 `js_content` 正文
- **原始抓取落盘：** [`sources/raw/wechat_embodied_chunk_replanning_53_papers_exrl_2026-11-11.md`](../raw/wechat_embodied_chunk_replanning_53_papers_exrl_2026-11-11.md)
- **核对范围：** 文章全文读完（53 条：1 篇深读 ExRL、2 篇跟进 LingBot-Touch 与 Embodied First Exam、50 篇扫读；每条为标题、一句话判断、结果说明加论文 / 代码 / 项目页链接）；文末评论未读；文章没有需要看的配图。53 篇论文里 43 篇新建节点，每篇由一个独立子任务通读论文（读不到的部分如实写在各节点「核对范围」），并逐句核对文章说法；其余 10 篇复用库里已有节点，**沿用既有页面，没有重新读论文**（其中 Compass-1 节点健康度为 2/3，本次未改动）。
- **通读：** 已完成
- **一句话说明：** 2026-11-11 更新的 53 篇具身智能新论文，按阅读优先级分三档；每篇论文对应唯一详情节点（新建 43 / 复用 10）。
- **说明：** 小站只给一句话判断和结果说明，且文章标题（动作块重规划）只对应第 1 篇 ExRL；其余多数论文与动作块无关（例如 LawKAN、Learning When to Think 的节点里都指出了这一点）。技术事实以各详情节点引用的 arXiv 原文为准。

## 53 篇 → 本库节点

| # | 档 | 论文节点 | arXiv | 状态 |
|---|----|----------|-------|------|
| 01 | 深读 | [长动作块，执行多久再重规划](../../wiki/entities/paper-exrl-execution-length-control.md) | [2611.13482](https://arxiv.org/abs/2611.13482) | 新建 |
| 02 | 跟进 | [换一种触觉传感器，表征能否复用](../../wiki/entities/paper-lingbot-touch.md) | [2611.12806](https://arxiv.org/abs/2611.12806) | 复用 |
| 03 | 跟进 | [会写代码之后，还要通过机器人实操题](../../wiki/entities/embodied-first-exam.md) | [2611.12177](https://arxiv.org/abs/2611.12177) | 复用 |
| 04 | 扫读 | [从动作生成先验，走向人形闭环操作](../../wiki/entities/paper-kimodo-policy.md) | [2611.11554](https://arxiv.org/abs/2611.11554) | 复用 |
| 05 | 扫读 | [把试错经验存进记忆，而非模型权重](../../wiki/entities/paper-world-in-context-learning-roboscript.md) | [2611.12885](https://arxiv.org/abs/2611.12885) | 复用 |
| 06 | 扫读 | [一句指令被误解，能否直接修正策略权重](../../wiki/entities/paper-weightsmith-instruction-distillation.md) | [2611.11894](https://arxiv.org/abs/2611.11894) | 新建 |
| 07 | 扫读 | [人在旁边动起来，操作策略还可靠吗](../../wiki/entities/paper-harp-bench-active-humans.md) | [2611.11123](https://arxiv.org/abs/2611.11123) | 新建 |
| 08 | 扫读 | [只用单目视频，导航如何记住走过的路](../../wiki/entities/paper-compass-1-navigation.md) | [2611.12087](https://arxiv.org/abs/2611.12087) | 复用 |
| 09 | 扫读 | [多位任务专家，怎样合成一套 VLA](../../wiki/entities/paper-vla-mopd.md) | [2611.12286](https://arxiv.org/abs/2611.12286) | 新建 |
| 10 | 扫读 | [世界动作模型扩量，先守住跨模态对齐](../../wiki/entities/paper-swam-cross-modal-alignment.md) | [2611.12870](https://arxiv.org/abs/2611.12870) | 复用 |
| 11 | 扫读 | [把世界模型的空间理解用于动作学习](../../wiki/entities/paper-foreact.md) | [2611.11500](https://arxiv.org/abs/2611.11500) | 新建 |
| 12 | 扫读 | [可通行区域分割，需要记忆也需要几何](../../wiki/entities/paper-sam-vtp.md) | [2611.12099](https://arxiv.org/abs/2611.12099) | 新建 |
| 13 | 扫读 | [检索旧示范，在不改权重时引导新动作](../../wiki/entities/paper-ragu-retrieval-augmented-guidance.md) | [2611.10894](https://arxiv.org/abs/2611.10894) | 新建 |
| 14 | 扫读 | [让系统辨识留下可检查的推理证据](../../wiki/entities/paper-aspire.md) | [2611.13114](https://arxiv.org/abs/2611.13114) | 新建 |
| 15 | 扫读 | [让潜在世界模型保留与动作有关的信息](../../wiki/entities/paper-sigma-lewm.md) | [2611.12732](https://arxiv.org/abs/2611.12732) | 新建 |
| 16 | 扫读 | [已知旋转组合规则，就把它写进模型](../../wiki/entities/paper-lawkan.md) | [2611.12567](https://arxiv.org/abs/2611.12567) | 新建 |
| 17 | 扫读 | [视觉策略异步运行，怎样让动作接得更顺](../../wiki/entities/paper-cac-async-act.md) | [2611.11089](https://arxiv.org/abs/2611.11089) | 新建 |
| 18 | 扫读 | [适配 VLA，验证损失未必选出最会动的模型](../../wiki/entities/paper-vla-adaptation-chess.md) | [2611.12676](https://arxiv.org/abs/2611.12676) | 复用 |
| 19 | 扫读 | [把第一视角视频和高精度全身动作对齐](../../wiki/entities/paper-robomocap.md) | [2611.10994](https://arxiv.org/abs/2611.10994) | 新建 |
| 20 | 扫读 | [世界模型候选更多，规划就一定更好吗](../../wiki/entities/paper-nearest-of-m-world-models.md) | [2611.11887](https://arxiv.org/abs/2611.11887) | 新建 |
| 21 | 扫读 | [机器人从环境状态中提出下一项任务](../../wiki/entities/paper-cafe-closed-loop-task-formulation.md) | [2611.12983](https://arxiv.org/abs/2611.12983) | 新建 |
| 22 | 扫读 | [自然语言车队调度，类型正确还不够](../../wiki/entities/paper-typed-plans-fleet-control.md) | [2611.12654](https://arxiv.org/abs/2611.12654) | 新建 |
| 23 | 扫读 | [边听边跳，人形机器人提前接住节拍](../../wiki/entities/paper-aheadance.md) | [2611.13370](https://arxiv.org/abs/2611.13370) | 新建 |
| 24 | 扫读 | [从网络视频联合预测手与物体的未来](../../wiki/entities/paper-grasping-the-future.md) | [2611.12879](https://arxiv.org/abs/2611.12879) | 新建 |
| 25 | 扫读 | [跨实验室成功率，先统一任务怎么抽样](../../wiki/entities/paper-stochastic-challenges-kallax.md) | [2611.12804](https://arxiv.org/abs/2611.12804) | 新建 |
| 26 | 扫读 | [双臂规划，把可执行条件带进每一步](../../wiki/entities/paper-bridging-planning-execution-bimanual.md) | [2611.12409](https://arxiv.org/abs/2611.12409) | 新建 |
| 27 | 扫读 | [让压力直接变成颜色，供操作策略读取](../../wiki/entities/paper-pctac.md) | [2611.12383](https://arxiv.org/abs/2611.12383) | 新建 |
| 28 | 扫读 | [长程人形导航，让规划与控制各按节奏运行](../../wiki/entities/paper-lhnav.md) | [2611.12136](https://arxiv.org/abs/2611.12136) | 复用 |
| 29 | 扫读 | [仅看外观不够，让 VLA 感知物体冷热](../../wiki/entities/paper-thermo-vla.md) | [2611.11981](https://arxiv.org/abs/2611.11981) | 新建 |
| 30 | 扫读 | [机器人翻物体，让代码智能体在试错中修策略](../../wiki/entities/paper-flip-anything-agentic-self-learning.md) | [2611.11941](https://arxiv.org/abs/2611.11941) | 复用 |
| 31 | 扫读 | [手术视频三维追踪，先把接触关系保留下来](../../wiki/entities/paper-surgtap-3d.md) | [2611.11697](https://arxiv.org/abs/2611.11697) | 新建 |
| 32 | 扫读 | [部署后的失败监测，持续更新最后一层](../../wiki/entities/paper-streaming-failure-forecasting.md) | [2611.11688](https://arxiv.org/abs/2611.11688) | 新建 |
| 33 | 扫读 | [物体被遮挡以后，世界模型还要预测整体运动](../../wiki/entities/paper-objectworld.md) | [2611.11085](https://arxiv.org/abs/2611.11085) | 新建 |
| 34 | 扫读 | [何时请人接管，要分清换阶段与真失败](../../wiki/entities/paper-moe-dagger.md) | [2611.10825](https://arxiv.org/abs/2611.10825) | 新建 |
| 35 | 扫读 | [简单状态少算几步，复杂状态再多想一会](../../wiki/entities/paper-aic-adaptive-internal-computation.md) | [2611.10822](https://arxiv.org/abs/2611.10822) | 新建 |
| 36 | 扫读 | [编程智能体接上机械臂，风险会怎样变化](../../wiki/entities/paper-armedagent-coding-agent-embodied-safety.md) | [2611.11438](https://arxiv.org/abs/2611.11438) | 新建 |
| 37 | 扫读 | [多设备共享世界状态，证据链也要一起评测](../../wiki/entities/paper-creg-evidence-grounded-shared-belief.md) | [2611.11842](https://arxiv.org/abs/2611.11842) | 新建 |
| 38 | 扫读 | [动作看起来要失败时，再触发仿真搜索](../../wiki/entities/paper-look-ahead-search-pusht.md) | [2611.10769](https://arxiv.org/abs/2611.10769) | 新建 |
| 39 | 扫读 | [越野车倾覆预警，把不同风险信号分开学](../../wiki/entities/paper-rover-tipover-multichannel.md) | [2611.12261](https://arxiv.org/abs/2611.12261) | 新建 |
| 40 | 扫读 | [大范围三维探索，把精细地图留在近处](../../wiki/entities/paper-flavor-hierarchical-spatio-temporal-map-exploration.md) | [2611.11489](https://arxiv.org/abs/2611.11489) | 新建 |
| 41 | 扫读 | [灵巧操作，把未来空间变化蒸馏进策略](../../wiki/entities/paper-dexforesight.md) | [2611.11166](https://arxiv.org/abs/2611.11166) | 新建 |
| 42 | 扫读 | [用激光几何约束，让高斯表示更贴近表面](../../wiki/entities/paper-pls-gs.md) | [2611.10699](https://arxiv.org/abs/2611.10699) | 新建 |
| 43 | 扫读 | [窄走廊导航，把定位和避障接到一起](../../wiki/entities/paper-maze-runner.md) | [2611.12949](https://arxiv.org/abs/2611.12949) | 新建 |
| 44 | 扫读 | [可穿戴机器人，把传感与助力模块解耦](../../wiki/entities/paper-exostack.md) | [2611.12306](https://arxiv.org/abs/2611.12306) | 复用 |
| 45 | 扫读 | [生成动作反复修改，也需要版本和验收记录](../../wiki/entities/paper-kinora.md) | [2611.11601](https://arxiv.org/abs/2611.11601) | 新建 |
| 46 | 扫读 | [多机器人搜救，把感知误差带进协同决策](../../wiki/entities/paper-pa-sar.md) | [2611.11648](https://arxiv.org/abs/2611.11648) | 新建 |
| 47 | 扫读 | [只靠局部本体感知，估计软体整体运动](../../wiki/entities/paper-deepsomo.md) | [2611.11242](https://arxiv.org/abs/2611.11242) | 新建 |
| 48 | 扫读 | [生成的人类动作，怎样变成机器人学习数据](../../wiki/entities/paper-dreamvla.md) | [2611.10872](https://arxiv.org/abs/2611.10872) | 新建 |
| 49 | 扫读 | [静脉压缩仿真，速度之外还要看误差](../../wiki/entities/paper-femoral-vein-compressibility-sim.md) | [2611.12818](https://arxiv.org/abs/2611.12818) | 新建 |
| 50 | 扫读 | [同一状态多采几种动作，保留示范多样性](../../wiki/entities/paper-vine-sampling-diffusion-policy.md) | [2611.10949](https://arxiv.org/abs/2611.10949) | 新建 |
| 51 | 扫读 | [从顶视相机起步，搭建小车编队实验](../../wiki/entities/paper-formation-control-tutorial.md) | [2611.11164](https://arxiv.org/abs/2611.11164) | 新建 |
| 52 | 扫读 | [把 ROS 2 轨迹搬进混合现实预览](../../wiki/entities/paper-ros2mr.md) | [2611.11830](https://arxiv.org/abs/2611.11830) | 新建 |
| 53 | 扫读 | [量子策略到机器人动作，中间链路如何追溯](../../wiki/entities/paper-qubit-to-motion.md) | [2611.11580](https://arxiv.org/abs/2611.11580) | 新建 |

## 逐篇核对中发现的、文章说法需要限定的地方（节选）

各节点「与文章说法的核对」一节有完整列表；下面是对读者最可能误读的几处。

- **ExRL**：「固定执行 10 步」只是表 I 里 1、10、50 三个固定长度中最好的一个；每个任务各用单独微调的 π0.5。
- **LawKAN**：与动作块无关；机械臂实验是公开数据集上的离线标定，FR3 上相对高斯牛顿只低约 0.2 mm。
- **Learning When to Think**：「停止推演」是在隐表示上反复精炼，不查询环境模型；论文没有动作块；多数回报增益很小。
- **ObjectWorld**：「刚体和形变物体」只在表示层面成立，真实标注靠刚体位姿跟踪；增益主要来自仿真。
- **Look Ahead**：只有单检查点单任务；项目页的数字与论文不一致。
- **Streaming Failure Forecasting**：没有在线真机运行；仿真只用 SmolVLA。
- **其他需留意的**：以各节点表格为准，本批中的 Zero-Shot VLA（RAGu）真机每任务仅 10 次，ASPIRE 的「封存测试集」只对智能体封存。
- **PLS-GS / FLAVOR / Vein 仿真**：文章的「实地」「实体仿体校准」等措辞偏宽，具体见各节点。
