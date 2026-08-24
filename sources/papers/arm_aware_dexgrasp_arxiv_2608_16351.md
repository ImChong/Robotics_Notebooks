# Arm-Aware Guided Dexterous Grasp Generation with Arm-Agnostic Grasp Models（arXiv:2608.16351）

> 来源归档（ingest）

- **标题：** Arm-Aware Guided Dexterous Grasp Generation with Arm-Agnostic Grasp Models
- **类型：** paper / dexterous-grasping / diffusion / inference-time-guidance
- **arXiv abs：** <https://arxiv.org/abs/2608.16351>
- **PDF：** <https://arxiv.org/pdf/2608.16351>
- **项目页：** <https://arm-aware-dexgrasp.github.io/>（归档见 [`sources/sites/arm-aware-dexgrasp-github-io.md`](../sites/arm-aware-dexgrasp-github-io.md)）
- **入库日期：** 2026-08-24
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md)

## 开源状态（步骤 2.5，2026-08-24）

- 匿名 RA-L 投稿项目页：方法视频与实机结果，**无 GitHub 链**。
- **结论：** **待发布**。

## 摘录 1：问题

- 悬浮手姿态扩散抓取模型忽略臂碰撞、工作空间边界与连续抓取效率。
- 拒绝采样在强约束下效率低；按臂重训限制泛化。

## 摘录 2：方法

- 复用预训练 **机械臂无关** 抓取扩散模型；推理时将臂运动学与 **环境 SDF** 作为约束梯度注入去噪。
- 联合优化手部位姿与臂构型；证明等价于 **引导扩散采样**。

## 摘录 3：评测

- 1 万物体 × 6 场景；走廊/货架等强约束环境可行抓取率显著高于拒绝采样；UR5 + LEAP Hand 实机。

**对 wiki 的映射：** [`wiki/entities/paper-arm-aware-dexgrasp.md`](../../wiki/entities/paper-arm-aware-dexgrasp.md)；交叉 [Manipulation](../../wiki/tasks/manipulation.md)。

## 当前提炼状态

- [x] 项目页核查（待发布）
- [x] 升格 `wiki/entities/paper-arm-aware-dexgrasp.md`
