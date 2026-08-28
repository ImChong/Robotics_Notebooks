# WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression

> 来源归档（ingest）

- **标题：** WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression
- **类型：** paper / next-scale autoregressive world model / policy evaluation / long-horizon video
- **PDF：** <https://github.com/X-Square-Robot/wall-ss/blob/main/wall-ss-paper.pdf>（仓库内 `wall-ss-paper.pdf`；页眉 `arXiv:submit/7998075 [cs.RO] 26 Aug 2026`，截至 **2026-08-28** 尚无公开 `arxiv.org/abs` 编号）
- **项目页：** <http://x2robot.com/pages/ss>（英文：<https://x2robot.com/en/pages/ss>）
- **代码：** <https://github.com/X-Square-Robot/wall-ss>
- **机构：** 自变量机器人（X Square Robot）
- **作者：** Maeve Zhang\*、Rain Sun\*、Xiang Wang\*、Cyril Zhang\*、Shalfun Li\*\*†、Meng Cao、Howard Lu、Ethan Chen 等；Hao Wang‡、Qian Wang（\* 核心贡献；† 项目负责人；‡ 通讯）
- **日期：** 2026-08-26（论文）；GitHub 2026-08-27
- **入库日期：** 2026-08-28
- **一句话说明：** 从 InfinityStar 初始化的 **下一尺度自回归** 具身世界模型：把观察–动作写成因果序列，用尺度对齐动作条件、有界时间–尺度记忆与 on-policy 视觉对齐，做动作可控、最长约 **60 s** 的流式推演，并用 **600** 组虚实配对验证策略评估校准。

## 开源状态（项目页 + 仓库核查，2026-08-28）

- **宣称将开源 / 训练推理代码待发布：** [X-Square-Robot/wall-ss](https://github.com/X-Square-Robot/wall-ss) 已公开（**MIT** LICENSE），但 README TODO 写明 *Release the training and inference code* **未勾选**；仓内仅 `README.md`、`LICENSE`、`assets/`、`wall-ss-paper.pdf`。项目页 <http://x2robot.com/pages/ss> 为前端展示，未列可运行训练/推理入口或权重。
- **勿写成已开源可复现：** 入库日无可辨识的 `train.py` / `eval.py` / checkpoint。

## 摘要级要点

- **瓶颈：** clip 级扩散 WM 把动作当全局条件，容易学「成功演示捷径」（磁铁式抓取）；长程自回归又会把误差写进后续上下文。
- **三件套：** (1) **尺度对齐动作条件** 贯穿 coarse→fine；(2) **尺度压缩长程记忆**（近细远粗 + 首帧锚点）+ **scale-wise dream forcing**；(3) 把 next-scale 视觉生成当随机策略，用动作跟随 / 长程一致性奖励做 **on-policy 对齐**（不优化任务成功率）。
- **数据：** 公开 [AgiBotWorld-Beta](https://arxiv.org/abs/2503.06669)（约 **98.8 万** captioned clip / **16.6 万** 源视频）+ ManipArena；私有 X2-Robot 双臂与 UMI；失败 / 接管 / rollback-replay 纠偏。动作条件只在标定链有效时启用，否则落入纯视频池。
- **评测：** WorldArena 风格 **200 ID + 100 OOD**；动作跟随 **0.290**、轨迹准确 **0.539**；虚实闭环 MAE **0.062**、\(r=0.93\)、组内排序 pairwise **0.89**；共训动作专家真机 Task Progress **69.1** vs π₀.₅ **49.6**。

## 核心论文摘录（MVP）

### 1) 流式动作因果 vs clip 级扩散

- **链接：** §1–§2；Fig. 2
- **摘录要点：** 常规视频扩散与动作条件扩散仍按 **未来 clip** 组织；WALL-SS 把任务、观察、动作写成 **时间交错因果序列**，每个未来观察由历史 + 中间动作生成，从而同时支持统一表征、可变长度、流式状态复用与 token 级似然（RL 接口）。
- **对 wiki 的映射：**
  - [WALL-SS](../../wiki/entities/paper-wall-ss.md) — 方法坐标。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 自回归 vs 扩散 WM。

### 2) 尺度对齐动作 + 有界记忆 + dream forcing

- **链接：** §3.2–§3.3；Eq. (1)–(5)
- **摘录要点：** 确定性渲染器把末端/夹爪计划投影到相机时间线（不用未来遥测）；动作 token 作 Transformer 条件前缀，按尺度读不粗于当前视觉尺度的动作。记忆复用同一 coarse-to-fine 层级：近期细、远期粗、首帧身份锚点；KV 预算不随 clip 数增长。Dream forcing 在损坏/自生成历史上预测干净未来，专治暴露偏差。
- **对 wiki 的映射：**
  - [WALL-SS](../../wiki/entities/paper-wall-ss.md) — 核心机制。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 长程像素仿真误差累积。

### 3) 虚实闭环校准与动作专家

- **链接：** §6.5–§6.6；Tab. 1、Tab. 3；Fig. 12–14
- **摘录要点：** 冻结外部策略（WALL-WM 五档 checkpoint）在 WM 与真机上各跑 **6 任务 × 20 初态 = 600** 对；校准 MAE **0.062**、\(r=0.93\)，组内 pairwise **0.89**，episode 平衡准确 **0.88**；乐观偏差集中在接触/插入。共训 flow-matching 动作专家读已提交因果状态，真机平均 Task Progress **69.1**。
- **对 wiki 的映射：**
  - [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) — 虚拟策略评估。
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 同属 policy-in-the-loop，骨干与开源状态不同。

## BibTeX

```bibtex
@article{wallss2026,
  title   = {WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression},
  author  = {Zhang, Maeve and Sun, Rain and Wang, Xiang and Zhang, Cyril and Li, Shalfun and Cao, Meng and Lu, Howard and Chen, Ethan and jhou, Harry and Zheng, KZ and Shi, Lights and Cheng, Regis and Lorenzin and Wang, Robert and Yao, Victor and Li, Gody and Mon, Elise and Tang, Yohann and Yu, Ryan and Zhang, PS and Chen, Vincent and Su, Hang and Gan, Roy and Wang, Hao and Wang, Qian},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-wall-ss.md`](../../wiki/entities/paper-wall-ss.md)
- 代码归档：[`sources/repos/wall-ss.md`](../repos/wall-ss.md)
- 项目页：[`sources/sites/x2robot-wall-ss.md`](../sites/x2robot-wall-ss.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[Ctrl-World](../../wiki/entities/paper-ctrl-world.md)、[OSCAR](../../wiki/entities/paper-oscar.md)、[Cosmos 3](../../wiki/entities/cosmos-3.md)
