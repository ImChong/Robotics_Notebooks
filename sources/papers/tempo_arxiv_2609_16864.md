# TEMPO: Learning Temporal Context for Dynamic Robot Manipulation

> 来源归档（ingest）

- **标题：** TEMPO: Learning Temporal Context for Dynamic Robot Manipulation
- **作者：** Zhenyang Feng、Jimin Heo、Erik B. Sudderth、Unnat Jain
- **机构：** 加州大学欧文分校（University of California, Irvine）
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.16864>
- **HTML 全文：** <https://arxiv.org/html/2609.16864v1>
- **项目页：** <https://tempo-robot.github.io/>
- **官方代码：** <https://github.com/tempo-robot/TEMPO>
- **发表信息：** arXiv:2609.16864v1，2026-09-15；arXiv 页面注明接收于 CoRL 2026
- **入库日期：** 2026-10-10
- **一句话说明：** 为单帧 VLA 补充视觉运动摘要（TEMPO-MOT）和本体感知动作历史（TEMPO-ACT），并发布用于运动感知评测的 TEMPO-Bench。

## 核心摘录（MVP）

### 1) 动态操作的两类表征失败

- **摘录要点：** 单帧观测看得到目标当下位置，却无法由静态画面确定目标如何运动，作者称其为 **motion ambiguity**；多阶段任务中相似画面可能分别出现在伸手接近和放下撤回阶段，要求相反动作，称为 **state aliasing**。论文据此认为，瓶颈主要是缺少时间上下文，而非单纯扩大模型或降低推理延迟。
- **对 wiki 的映射：**
  - [TEMPO 动态操作](../../wiki/entities/paper-tempo-dynamic-manipulation.md) — 问题定义与方法动机。
  - [VLA](../../wiki/methods/vla.md) — 单帧策略与动态操作的时间上下文边界。

### 2) MOT 与 ACT 两种互补输入

- **摘录要点：** TEMPO-MOT 将当前视觉特征作为 query，对滚动缓存的历史帧特征进行 cross-attention；默认视觉编码器为冻结的 SAM 2.1-Tiny。TEMPO-ACT 将近期本体感知命令切分为固定数量时间桶并取每桶均值，论文设置为 10 桶、每桶 30 条命令。两路特征接入 VLM 前缀，ACT 还经 AdaRMS 残差调制 action expert；增加约 2M 参数（0.08%），不修改预训练骨干。
- **对 wiki 的映射：**
  - [TEMPO 动态操作](../../wiki/entities/paper-tempo-dynamic-manipulation.md) — 架构与方法流程。
  - [Action Chunking](../../wiki/methods/action-chunking.md) — 输出动作块与时序控制的关系。

### 3) 真机结果与异步推理的互补性

- **摘录要点：** 双臂平台上，Bottle Handover 从 VLASH 的 38% 提升到 74%；完整 Flick Catch 与 Wine Pour 中 RTC、VLASH 均为 0%，TEMPO 分别为 68% 与 97.6%。主结果为和基线进行可比评估，将这两项任务的 release-and-retract 状态混淆片段裁剪；其主表 TEMPO 数值为 66% 和 98.1%。Drop Catch 更受执行时序影响，RTC 在该任务更强。论文报告 RTX PRO 6000 上策略中位前向 35.3 ms，对比 VLASH 33.8 ms，MOT 编码并行处理。
- **对 wiki 的映射：**
  - [TEMPO 动态操作](../../wiki/entities/paper-tempo-dynamic-manipulation.md) — 评测边界与时延读法。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 动态操作的跟踪与阶段判断子问题。

### 4) TEMPO-Bench 与公开复现边界

- **摘录要点：** 论文称 TEMPO-Bench 包含超过 50,000 帧的人机动态操作视频，带逐帧物体速度标注，并提供运动方向/幅度的多选版本。GitHub README 提供基于 LeRobot 的数据预处理、PyTorch 微调与策略服务示例；但同一 README 将训练数据、训练权重和 YAM 部署列在 TODO，且没有给出 TEMPO-Bench 下载地址。仓库代码公开不等于论文数据和预训练检查点已公开。
- **对 wiki 的映射：**
  - [TEMPO 动态操作](../../wiki/entities/paper-tempo-dynamic-manipulation.md) — TEMPO-Bench 与数据开放状态。
  - [TEMPO 官方仓库归档](../repos/tempo-robot-tempo.md) — 可运行代码和待发布内容。

## 当前提炼状态

- [x] arXiv HTML 全文中的方法、实验和消融与摘要对齐
- [x] 核对官方项目页、官方仓库与公开状态
- [x] Wiki 映射：新增 TEMPO 动态操作实体，并连接 VLA 与 Manipulation
