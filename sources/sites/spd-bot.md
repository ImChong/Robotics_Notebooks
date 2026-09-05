# spd.bot（SPD 项目页）

- **标题：** Pre-training Visual Dexterity in Simulation
- **类型：** site / project-page
- **URL：** <https://spd.bot/>
- **PDF（项目页托管）：** <https://spd.bot/assets/paper.pdf>
- **配套论文归档：** [`sources/papers/spd_corl_2026.md`](../papers/spd_corl_2026.md)
- **机构：** Stanford University¹ · MIT² · Scale AI³（\* 同等贡献）
- **作者：** Sarthak Kamat¹\*、Adam Rashid²\*、Satvik Sharma¹、Aseem Doriwala³、Chelsea Finn¹、Phillip Isola²、C. Karen Liu¹
- **入库日期：** 2026-08-17
- **复核日期：** 2026-09-05（补挂 arXiv:2608.15917；代码/数据仍无 URL）

## 一句话摘要

斯坦福 / MIT / Scale AI 的 CoRL 2026 项目页：用 VR 头显在仿真里遥操双臂灵巧手，5 人一周采 **75 小时** 演示（spd-75h），预训练扩散 Transformer 后，真机每任务 **1–2 小时** 微调即可完成放盘子、挂马克杯、叠叠乐等五项接触丰富任务，且五项都胜过从零 BC。

## 开源状态（步骤 2.5，截至 2026-09-05）

打开 <https://spd.bot/> 与 <https://arxiv.org/abs/2608.15917> 核头部按钮与页脚资源区：

| 资源 | 状态 |
|------|------|
| 项目页 / 论文 PDF | **已发布**（`assets/paper.pdf`；PDF Producer 标注 *Proceedings of the 10th Conference on Robot Learning (CoRL 2026)*） |
| arXiv | **已发布** — [arXiv:2608.15917](https://arxiv.org/abs/2608.15917)（项目页 BibTeX 已更新） |
| 代码 / spd-vr | **未列 URL**（论文宣称释放 VR 遥操作软件与六套调参场景） |
| 数据集 spd-75h | **未列 URL**（论文宣称释放；致谢 Scale AI 采集） |
| 权重 / Hugging Face / GitHub | **无链接** |

**结论：宣称将开源 / 待核实。** 复现入口暂以项目页 PDF 与叙述为准；wiki「源码运行时序图」标 **不适用**，待正式 release 后补 `sources/repos/`。

## 公开信息要点

- **卖点：** 灵巧手预训练数据不必上真机；仿真采集可瞬间 reset、并行、异地。
- **采集：** 操作员戴 VR 头显控虚拟双臂灵巧手；头显跟踪人手，经 IK 驱动臂与手指；接触全部物理仿真。
- **规模：** 5 名操作员、约一周、**75 小时**、六场景、数百物体实例；任务开放结局、不锁死子任务顺序。
- **微调：** 与仿真对齐的真机遥操作，每任务 **1–2 小时** 演示，全策略微调。
- **模型：** **222M** 扩散 Transformer；无语言标注，条件于 **sensorimotor 历史**；256 步 × 30 Hz（约 8 s）；滑窗注意力 + 部署 rolling KV cache；短 **8 步** chunk、30 Hz 反应控制。
- **真机任务视频：** plate racking / cup stacking / Jenga / mug hanging / bottles in bin。
- **主结论：** 五项任务平均进度均高于同架构从零 BC；训练 loss 起点与收敛更低。
- **消融：** 单帧 + 短 chunk 崩；单帧 + 长 chunk 接近 π0 风格；**32 步历史 + 8 步 chunk** 最强，且从此配置预训练收益最大。
- **BibTeX：** `@article{spd2026, title={Pre-training Visual Dexterity in Simulation}, ... journal={arXiv preprint arXiv:2608.15917}, url={https://arxiv.org/abs/2608.15917}, year={2026}}`。

## 为何值得保留

- 把「灵巧手缺数据」从真机遥操作瓶颈改写成 **仿真 on-embodiment 演示可规模化** 的工程坐标。
- 与 [EgoScale](../../wiki/methods/egoscale.md)（人视频 off-embodiment）和 [TeleDexter](../../wiki/entities/paper-teledexter.md)（真机 co-tracking 采数）形成预训练数据源三角。
- 给 [Action Chunking](../../wiki/methods/action-chunking.md) 补一条真机证据：**历史条件化才能把 chunk 缩短到反应级**。

## 关联资料

- 论文归档：[`sources/papers/spd_corl_2026.md`](../papers/spd_corl_2026.md)
- wiki 实体：[`wiki/entities/paper-spd.md`](../../wiki/entities/paper-spd.md)
