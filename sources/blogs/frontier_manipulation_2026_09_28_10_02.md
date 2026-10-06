# 【9.28–10.2 前沿论文动态】Manipulation

- **作者：** senlanke（具身运控lab）
- **发表日期：** 2026-10-06
- **原文：** <https://mp.weixin.qq.com/s/QnO4PnLUoCM8olHahf8E6g>
- **归档依据：** 用户提供的 PDF《【9.28–10.2 前沿论文动态】Manipulation》。此页提炼文章条目与独立详情节点映射；论文技术事实仍以 arXiv 原文为准。
- **条目数：** 8 个论文条目；同一 arXiv 论文只保留一个 wiki 详情节点。

## 条目与详情节点

### Rho: A Foundation for Efficiently Adaptable VLA Models（arXiv:2609.38164）

- **论文：** [arXiv:2609.38164](https://arxiv.org/abs/2609.38164)
- **问题：** 通用 VLA 一方面需要从大规模数据获得跨任务能力，另一方面部署到具体机器人后又需 要低成本适应。现有方案通常需要针对目标本体重新进行大量 Fine-Tuning，机器人上线后遇到训练分布边 缘的新情况时，也很难通过少量人工纠正快速更新
- **方法线索：** Rho 是一组 5B 参数、开放权重的双臂 VLA。作者把“通用预训练 → embodiment midtraining → task adaptation”明确拆成几个阶段：先得到 Rho-base，再针对 YAM Box、UR AI Trainer、FR3 Duo 三种双 臂机器人进行本体级中间训练，最后使用少量任务数据适配。更有意思的是它的在线适应方式：不修改冻 结的 Flow-Matching Action Expert，而是训练一个轻量 latent policy，根据当前 observation 选择输入 Action Expert 的 noise。 只使用约 15 条人工纠正 episode，就可以把策略推向原离线 Fine-Tuning 分布之外 的任务状态。基础模型、三个本体 Checkpoint 和数据均已公开
- **详情节点：** [Rho：面向高效适应的 VLA 基础模型](../../wiki/entities/paper-rho.md)

### DSDyn-VLA: A Dual-Stream Dynamic Manipulation Framework with Motion Perception, Future Awareness, and Realtime Correction（arXiv:2609.39198）

- **论文：** [arXiv:2609.39198](https://arxiv.org/abs/2609.39198)
- **问题：** 现有 VLA 在静态桌面操作上已经很强，但遇到传送带、运动目标等动态场景会同时遇到 三个问题：单帧视觉缺乏运动信息；大型 VLA 推理延迟导致动作输出时目标已经移动；Action Chunk 一旦 生成后通常开环执行，执行期间无法根据最新状态及时纠正
- **方法线索：** DSDyn-VLA 把策略拆成 Slow-Fast 双流。慢速 Flow-Planner 负责宏观动作规划，在 VLA 中 显式加入 Optical Flow 运动信息，并通过 Future State Awareness 提前预测推理延迟后的状态，再生成 Action Chunk；快速 Res-Refiner 则是一个轻量 RL Policy，根据实时观测给已经规划好的动作持续叠加高频 Residual Correction。也就是说，它不是让大 VLA 自己提高控制频率，而是形成：低频 VLA Action Chunk + 高频 RL Residual。在高延迟 Kinetix 条件下失败率相对当前 SOTA 降低超过 **76%**；真实动态操作成功率 约为 π0.5 的 6 倍，DynBench 上约为 5 倍。作者同时构建了 9 个动态操作任务的 DynBench，并计划开放代 码和权重
- **详情节点：** [DSDyn-VLA：具有运动感知、未来感知与实时修正的双流动态操作框架](../../wiki/entities/paper-dsdyn-vla.md)

### GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives（arXiv:2609.39601）

- **论文：** [arXiv:2609.39601](https://arxiv.org/abs/2609.39601)
- **问题：** VLA 和 WAM 大多直接继承通用 VLM 或视频生成模型的视觉 Backbone，但机器人操作 对 Grounding 的要求比普通视觉问答高得多：不仅要知道“杯子在哪里”，还要准确定位很小的操作区域、接 触位置以及拥挤场景中的目标，并满足闭环控制的实时要求
- **方法线索：** GroundingPI 没有直接继续扩大 VLA，而是专门训练一个 4B Grounding Foundation Model， 把 point 和 bounding box 都量化成共享 vocabulary 中的坐标 token，通过 multimodal/spatial pretraining、SFT 和 GRPO 训练。在 34 个 Grounding Benchmark 上平均达到 73.68%。作为机器人视觉 Backbone 接入 RoboTwin 2.0 后，在四种 OOD 条件下均超过论文比较的主流 Backbone，最大相对提升 24.8%；在 RoboCasa-GR1 上只使用 50% demonstration，即超过部分使用 75% demonstration 的视觉 Backbone。它 体现的是另一条 VLA 路线：先把操作所需的精确物理 Grounding 做强，再接 Action Policy。
- **详情节点：** [GroundingPI：基于视觉基元的物理智能 Grounding 基础模型 机 构 ： XPeng Inc. 、 Peking University 、 The University of Hong Kong 、 UC Berkeley 、 Princeton University、NUS、Tsinghua University、HKUST (GZ) 等](../../wiki/entities/paper-groundingpi.md)

### RoboCoach: World Models as Active Coaches for Compositional Robot Skills（arXiv:2609.39685）

- **论文：** [arXiv:2609.39685](https://arxiv.org/abs/2609.39685)
- **问题：** 长程操作策略失败以后，通常继续收集整条任务的 End-to-End Demonstration，但真正的 问题往往只发生在其中某一个子技能。例如“开抽屉→拿物体→放入容器”失败，可能只需要补“抓取”阶段的 数据，重新收集整条轨迹会浪费大量真机数据
- **方法线索：** RoboCoach 让 World Model 不只是预测未来，而是承担 Active Coach （主动教练）。提出 RIDI： Route → Imagine → Diagnose → Improve。先在共享 Action-Conditioned World Model CoachWorld 中 运行各个 Skill Expert，想象整个任务执行过程；Progress Judge 找出第一个失败的 Subtask；系统统计大量 imagined failure 后，自动决定“下一批 demonstration 应该采哪个技能”以及“应该更新哪个 Expert Adapter”。 只增加 150 条 Subtask Demonstration，Franka 长程任务成功率从 **13.3% → 75.0%**，AgileX 从 40.0% → 83.8%。这篇工作的核心变化是： World Model 开始决定真实机器人下一步应该采什么数据，而不仅仅给 Policy 提供未来预测。
- **详情节点：** [RoboCoach：将世界模型作为组合式机器人技能的主动教练](../../wiki/entities/paper-robocoach.md)

### Tactile Curiosity Drives Robot Interaction（arXiv:2609.40134）

- **论文：** [arXiv:2609.40134](https://arxiv.org/abs/2609.40134)
- **问题：** 机器人随机探索会浪费大量动作在自由空间；普通不确定性奖励也可能偏好自由空间中的模型未知，而不是能改变抓取、滑动和稳定性的接触经验。
- **方法线索：** TacEx 按视觉与触觉模态分解认知不确定性，并提高触觉不确定性在内在奖励中的权重，引导策略主动探索陌生接触。探索无需任务奖励或专家示范；收集的交互回放冻结后再补标任务奖励，用于离线强化学习和触觉后训练。
- **详情节点：** [触觉好奇心驱动机器人交互](../../wiki/entities/paper-tactile-curiosity-drives-robot-interaction.md)

### Counterfactual Video Generation Enables Scalable Humanoid Loco- Manipulation（arXiv:2609.38172）

- **论文：** [arXiv:2609.38172](https://arxiv.org/abs/2609.38172)
- **问题：** 从人类视频学习 Humanoid Loco-Manipulation 很有吸引力，但真正适合训练的数据很难 采：视频既要看清完整人体运动，又要看清人与物体的接触，遮挡不能太严重，同时还需要覆盖不同人 体、物体和运动方式。仅靠人工拍摄高质量 HOI 视频很难扩大规模
- **方法线索：** 提出 PRISM，不是不断拍摄更多真实视频，而是从少量 exemplar video 出发，通过 Video-to- Video Generation 生成数百条不同的 counterfactual human-object interaction video；随后使用 contact- anchored Real-to-Sim Pipeline 重建人体和物体运动，并把生成视频中并不完全物理正确的运动重新投射成仿 真中可执行的 physically plausible trajectory，再进行 Humanoid Policy Learning 和 Sim-to-Real。核心数据路线 变成：少量真实人类视频 → 大量生成 HOI 视频 → 物理重建 → Humanoid Policy。这篇的重点不是让视频生 成模型直接控制机器人，而是把生成式视频作为 Humanoid Manipulation 数据放大器。
- **详情节点：** [反事实视频生成实现可扩展人形机器人移动操作](../../wiki/entities/paper-prism-real2sim2real.md)

### T²Mem: Learning Test-Time Memory for Robotics（arXiv:2609.36720）

- **论文：** [arXiv:2609.36720](https://arxiv.org/abs/2609.36720)
- **问题：** 很多长程 Manipulation 是部分可观测的。例如机器人几分钟前看见某个物体被放进抽屉， 现在当前相机画面已经没有这条信息；仅扩大 observation-history window 会不断增加 Transformer 的上下文 长度和推理开销，而且保存“所有历史”并不等于保存“未来真正需要的信息”
- **方法线索：** T²Mem 不增加独立 Memory Model，也不要求额外 Memory Annotation，而是直接让预训练 VLA 自己在 Test Time 学习记忆。历史 observation 通过在线 self-supervised update 被写入一组 compact fast weights，之后 VLA 不需要每一步重新处理全部历史，而是从 fast weights 中读取历史信息。也就是说， Memory 不再只是 Context Token，而是变成推理过程中持续更新的一小部分参数状态。训练监督仍来自普通 Action Demonstration，不需要额外标注“应该记住什么”。
- **详情节点：** [T²Mem：面向机器人的测试时记忆学习](../../wiki/entities/paper-t2mem.md)

### SafeVLA-Bench: A Benchmark for the Success-Safety Gap in Vision- Language-Action Models（arXiv:2606.00773）

- **论文：** [arXiv:2606.00773](https://arxiv.org/abs/2606.00773)
- **问题：** 当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让抓取物体失稳，甚至 发生机器人 self-contact，这些通常不会反映在最终 success rate 中
- **方法线索：** SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 rollout 上进行 post-hoc safety evaluation，不需要重新设计整个 Benchmark。除了 Success，还报告 Safety Rate 、 Success-but-Unsafe Rate，以及描述最严重违规程度的 Violation Severity Index。实验发现，即使平均成功率超过 90% 的 15 个 tabletop policy，仍有 18–28% unsafe episode； RoboCasa-365 中 38–56% 的成功 rollout 至少违反一个安全条件。论文还展示了利用这些指标进行 Policy Post-Training 的案例。 【9.21-9.25前沿论文动态】Manipulation15 篇
- **详情节点：** [SafeVLA-Bench：视觉-语言-动作模型成功率与安全性差距评测基准](../../wiki/entities/paper-safevla-bench.md)
