# 机器人论文密集上新：从 Ego2Robot 到 MANGO-Grasp，下一轮竞争焦点变了

> 来源归档（blog / 微信公众号）

- **标题：** 机器人论文密集上新：从 Ego2Robot 到 MANGO-Grasp，下一轮竞争焦点变了
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/nKF7rxH-OuJz68galP3Xpg
- **发表日期：** 2026-08-11（frontmatter；文内日期 2026-08-10）
- **入库日期：** 2026-08-15
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_ego2robot_mango_grasp_2026-08-11.md`](../raw/wechat_embodied_station_ego2robot_mango_grasp_2026-08-11.md)
- **一句话说明：** 汇总 9 篇近期具身/机器人论文（均宣称开源或有项目页），主线从「堆模型规模」转向 **数据来源、表征粒度与控制接口** 的可迁移性；本库 **复用 2 个已有完整节点，新建 7 个独立论文实体页，不重复造页**。

## 核心摘录（归纳，非全文）

文内判断：这批工作把「可泛化」写进接口——pose geometry、physical brush、anatomical unit、embodied latent、semantic re-binding、morpho-kinematic descriptors。下一轮竞争不在「大模型更大」，而在数据、表征和控制闭环能否跨场景 / 任务 / 本体复用。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | PFM-HR | [2608.03227](https://arxiv.org/abs/2608.03227) | 仓 Coming Soon | **复用** [paper-pfm-hr](../../wiki/entities/paper-pfm-hr.md) |
| 02 | OnOff（可微物理笔刷） | [2608.03198](https://arxiv.org/abs/2608.03198) | 项目页已发，代码未列 | [paper-onoff-handwriting](../../wiki/entities/paper-onoff-handwriting.md) |
| 03 | DigitCode | [2608.03127](https://arxiv.org/abs/2608.03127) | 演示页已发，HandTok 待审稿后挂 | [paper-digitcode](../../wiki/entities/paper-digitcode.md) |
| 04 | EmbodiedVAE | [2608.02990](https://arxiv.org/abs/2608.02990) | 仓 Coming Soon | [paper-embodiedvae](../../wiki/entities/paper-embodiedvae.md) |
| 05 | Ego2Robot | [2608.02580](https://arxiv.org/abs/2608.02580) | 项目页已发，管线/数据未开 | [paper-ego2robot](../../wiki/entities/paper-ego2robot.md) |
| 06 | Situation-aware Frontier | [2608.02571](https://arxiv.org/abs/2608.02571) | **已开源** `go2_rescue_eval` | [paper-situation-aware-frontier-quadruped-sar](../../wiki/entities/paper-situation-aware-frontier-quadruped-sar.md) |
| 07 | Why Action Chunking Improves BC | [2608.02547](https://arxiv.org/abs/2608.02547) | 项目页 PDF；代码 Coming soon | **复用** [paper-why-action-chunking-improves-bc](../../wiki/entities/paper-why-action-chunking-improves-bc.md) |
| 08 | GSR / ParaVLA | [2608.02497](https://arxiv.org/abs/2608.02497) | **已开源** 训练/评测 + HF ckpt | [paper-gsr-paravla](../../wiki/entities/paper-gsr-paravla.md) |
| 09 | MANGO-Grasp | [2608.02014](https://arxiv.org/abs/2608.02014) | 宣称出版后开源 | [paper-mango-grasp](../../wiki/entities/paper-mango-grasp.md) |

### 文内要点速记

1. **PFM-HR** — 无序 pose 上训 Flow Matching 先验，用 Pose Geometry Score 调制跟踪奖励。
2. **OnOff** — 六参数可微笔刷打通 online 轨迹与 offline 图像，Lite 6 真机书法。
3. **DigitCode** — 手部 token 粒度落到骨 / 指 / 整手；量化误差约降 3/4。
4. **EmbodiedVAE** — 双编码器解耦机械臂运动与背景，服务操作世界模型。
5. **Ego2Robot** — 第一人称人视频 → 15 形态、18,561 h 机器人训练数据。
6. **Frontier SAR** — 四足搜救 frontier 排序加入救援相关性，复杂 clutter 下完成率最高。
7. **Action Chunking** — 收益主因是延迟条件化 + 隐式集成，而非「必须播整段 chunk」。
8. **GSR** — VLA 改写指令崩溃来自 joint routing，不是语义不懂；LIBERO-Para 最高 +44.6 pp。
9. **MANGO-Grasp** — 几何 3DGS + 形态–运动学描述子；未见 SharpaWave 零样本，真机 86%。

## 对 wiki 的映射

- **不新建** PFM-HR / Why Action Chunking 实体：二者已是 `status: complete`。
- **新建 7** 个独立 `paper-*` 详情节点；各页 `参考来源` 回链本博客。
- 交叉：[VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[EgoScale](../../wiki/methods/egoscale.md)、[WiLoR](../../wiki/methods/wilor.md)、[RoboTwin](../../wiki/entities/robotwin.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[抓取位姿估计](../../wiki/methods/grasp-pose-estimation.md)、[UHAS](../../wiki/methods/uhas-unified-hand-action-space.md)、[世界动作模型](../../wiki/concepts/world-action-models.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（2 复用 / 7 新建 / 0 stub 重复）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
