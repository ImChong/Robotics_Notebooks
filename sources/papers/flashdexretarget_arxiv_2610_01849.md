# FlashDexRetarget（arXiv:2610.01849）

> 来源归档（ingest）

- **标题：** FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting
- **缩写：** **FlashDexRetarget**
- **类型：** paper / dexterous manipulation / human-object interaction / multi-reference retargeting / reinforcement learning
- **arXiv：** <https://arxiv.org/abs/2610.01849>
- **PDF：** <https://arxiv.org/pdf/2610.01849>
- **HTML（v2）：** <https://arxiv.org/html/2610.01849v2>
- **项目页：** <https://holiday-robot.github.io/FlashDexRetarget/> — [开放状态核查](../sites/flashdexretarget-project.md)
- **代码仓库：** <https://github.com/DAVIAN-Robotics/FlashDexRetarget> — [仓库核查](../repos/flashdexretarget.md)
- **作者：** Kyungmin Lee、Sibeen Kim、Dongyoon Hwang、Yoonsang Oh、Donghu Kim、Youngdo Lee、I Made Aswin Nahrendra、Jaegul Choo、Hojoon Lee
- **机构：** KAIST AI；Holiday Robotics
- **版本：** v1 于 2026-10-01 提交；当前核查版本 v2，2026-10-04 修订
- **入库日期：** 2026-10-07
- **开源状态：** 项目页的 Code 按钮标注 coming soon；官方 GitHub 仓库 README 说明算法代码将发布在该仓库，目前可见内容用于托管项目页。尚无可运行训练实现或项目权重/数据发布链接。

## 核心论文摘录（MVP）

### 1) 多参考策略摊销逐动作优化

- **链接：** <https://arxiv.org/abs/2610.01849>
- **核心贡献：** 不再为每段演示分别训练策略，而是把整个演示集合建模成 multi-reference tracking，由一个 reference-conditioned policy 联合训练。并行仿真从集合中抽样参考轨迹，使更新经验可服务多段动作。
- **对 wiki 的映射：** [FlashDexRetarget 实体页](../../wiki/entities/paper-flashdexretarget.md)；[动作重定向概念](../../wiki/concepts/motion-retargeting.md)

### 2) 用几何与未来动作编码识别交互

- **核心贡献：** 当前和参考物体以手腕局部坐标系下的 128 个表面点表示；输入指尖及手腕到物体表面的距离和法向。策略还接收未来 (K=10) 帧的手—物参考，由时序编码器压为 128 维 latent。
- **对 wiki 的映射：** [FlashDexRetarget 方法流程](../../wiki/entities/paper-flashdexretarget.md#方法流程)；可与 [HOI-Retarget](../../wiki/entities/paper-hoi-retarget.md) 的接触中心时间窗优化比较。

### 3) 双手专属 actor-critic 与离策略学习

- **核心贡献：** 每只手使用独立 actor-critic 和手专属奖励；两者共享双手状态观测。使用 FlashSAC，扩大 replay buffer 至 50M transitions、critic 隐层至 1024，以重用多参考分布中的经验并提升 critic 表达能力。
- **对 wiki 的映射：** [FlashDexRetarget 方法](../../wiki/entities/paper-flashdexretarget.md#核心机制)；[SPIDER](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md) 代表物理采样式基线。

### 4) 50 动作基准与计算预算

- **核心结果（arXiv v2 Table I）：** TACO、OakInk2、HOT3D 的 50 段动作含 25 段单物体与 25 段双物体。XHand 上 SR_SPIDER 为 90%、严格 SR_MT 为 86%，用 29 GPU-hours；CHORD 为 46% / 0%，用 2,847 GPU-hours。Sharpa Wave Hand 上为 86% / 80%、44 GPU-hours；CHORD 为 50% / 0%、3,314 GPU-hours。
- **评测口径：** SR_SPIDER 按活动手物体位置/旋转的宽松阈值判成功；SR_MT 同时加入指尖与手关节跟踪阈值。v2 摘要把 XHand 对照概括为约 90% 对 46%、约 30 对 3,000 GPU-hours、约 100 倍更少计算量。
- **对 wiki 的映射：** [FlashDexRetarget 评测与结论](../../wiki/entities/paper-flashdexretarget.md#评测与结果)。

### 5) 规模扩展和局限

- **核心贡献：** 对 200、500、1,000 段参考演示进行共享训练；真实机器人回放演示擦板、向平底锅倒入及关盖。作者同时指出依赖较干净的手—物轨迹、准确物体几何和仿真特权信息；未来方向包括未见动作泛化与面向真实观测的 student policy。
- **对 wiki 的映射：** [FlashDexRetarget 局限与风险](../../wiki/entities/paper-flashdexretarget.md#局限与风险)。

## 可用资源

- [arXiv 论文与版本记录](https://arxiv.org/abs/2610.01849)
- [项目主页](https://holiday-robot.github.io/FlashDexRetarget/)
- [补充视频](https://holiday-robot.github.io/FlashDexRetarget/static/videos/flashdexretarget_supp.mp4)
- **源码：** 待发布；状态见[官方仓库归档](../repos/flashdexretarget.md)。

