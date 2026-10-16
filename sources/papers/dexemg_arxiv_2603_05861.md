# DexEMG: Towards Dexterous Teleoperation System via EMG2Pose Generalization

> 来源归档（ingest）

- **标题：** DexEMG: Towards Dexterous Teleoperation System via EMG2Pose Generalization
- **简称：** DexEMG
- **类型：** paper / dexterous-teleoperation / sEMG / hand-pose-estimation / motion-retargeting
- **arXiv abs：** <https://arxiv.org/abs/2603.05861>
- **PDF：** <https://arxiv.org/pdf/2603.05861>
- **v1 日期：** 2026-03-06（arXiv 摘要页提交历史；截至 2026-10-16 仅 v1）
- **作者：** Qianyou Zhao、Wenqiao Li、Chiyu Wang、Kaifeng Zhang（作者名单取自 arXiv 摘要页）
- **机构：** arXiv 摘要页不显示机构；论文 PDF 首页署名为 Sharpa（四位作者均有）与上海交通大学（SJTU，Zhao、Li 双署名）
- **arXiv 备注：** 摘要页未填写；分类 cs.RO
- **项目页：** 未找到（arXiv 摘要页未列；Sharpa 研究博客有一篇解读页，只回链 arXiv，不是项目页）
- **代码 / 数据：** 未找到（arXiv 摘要页、论文 PDF、Sharpa 解读页均未给出代码或数据链接；检索 Sharpa 官方 GitHub 组织未见 DexEMG / EMG 相关仓库）
- **入库日期：** 2026-10-16
- **一句话说明：** 用商用 8 通道 sEMG 臂带加腕部追踪器取代外骨骼与视觉动捕：先用动捕手套采同步的 sEMG 与手姿数据训练 EMG2Pose（TDS 编码器 + LSTM 解码器预测关节速度），再把预测的 22 DoF 手部动作实时驱动 Sharpa Wave 灵巧手。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv 摘要页 | <https://arxiv.org/abs/2603.05861> | 标题、作者、v1 日期、摘要 |
| Sharpa 研究博客解读 | <https://www.sharpa.com/blogs/research/breaking-the-barriers-of-teleoperation-how-dexemg-and-sharpa-wave-are-revolutionizing-dexterous-manipulation> | 厂商二手解读，只链 arXiv；该文把 MAE 0.09 rad 写成灵巧手的精度，论文原文是 EMG2Pose 网络的关节角误差，使用时以论文为准 |
| 同来源文章 | [机器之心 / Sharpa 微信文章归档](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md) | 文中在「主线一」下用一小段提及 DexEMG |

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 项目页 | **无**：arXiv 摘要页与论文 PDF 均未列 |
| 官方代码 | **未找到**：arXiv、论文、Sharpa 解读页均无链接；Sharpa 官方 GitHub 组织（18 个公开仓库）中无 DexEMG / EMG 相关仓库 |
| 数据 / 权重 | 未找到；论文正文也未给出采集时长、受试者数量或数据集发布计划 |
| 综合判定 | **截至 2026-10-16 未开源，论文亦未承诺发布** |

## 文章转述（微信文章，机器之心 2026-10-16）

> 以下为对文章原文的转述，标注「文章转述」，非论文原文。

- **文章转述：** DexEMG 属于文章「主线一：击穿具身鸿沟，让人类数据成为动作燃料」，与 UVTA、Dex-X 并列；文章称它「进一步降低采集门槛」。
- **文章转述：** 做法是用前臂的表面肌电信号推断手部姿态，再实时重定向到 22 自由度的 Sharpa Wave 灵巧手上，替代笨重的外骨骼和视觉动捕。
- **文章转述：** 由于不依赖摄像头，手指被物体遮挡时依然可以正常遥操作。
- **文章转述：** 文章把 UVTA 的真机触觉增强、DexEMG 降低采集门槛、Dex-X 的仿真力学补全并称，认为主线一正把数据采集重心从「逼迫人适应机器」转向「让系统理解人类的手感」。文章没有给出 DexEMG 的任何数字。

## 核心摘录（归纳，非全文）

1. **arXiv 摘要：** 遥操作系统常在性能与便携之间取舍——视觉捕捉受成本与视线限制，机械外骨骼笨重且限制活动。DexEMG 是一个轻量、低成本的表面肌电（sEMG）遥操作系统，用 sEMG 连接人的意图与机器人执行。
2. **arXiv 摘要：** 先用动捕手套采集同步的 sEMG 与手姿数据，训练能从肌肉活动连续预测手部运动学的神经网络 EMG2Pose；再配一个鲁棒的手部重定向算法，把预测姿态实时映射到多指灵巧手上。
3. **arXiv 摘要：** 实验称系统在多种遥操作任务中精度高，并能在新物体与复杂环境中泛化，且无需针对个人做密集重新标定。
4. **文章转述：** 「用前臂的表面肌电信号推断手部姿态，再实时重定向到 22 自由度的 Sharpa Wave 灵巧手上，替代笨重的外骨骼和视觉动捕。」
5. **文章转述：** 「由于不依赖摄像头，手指被物体遮挡时依然可以正常遥操作。」

## 论文正文要点（PDF v1，自行阅读；数字均为论文自报）

> 此节取自论文 PDF 正文，不属于上面三类摘录来源，单列便于核对。

- **硬件：** gForce 8 通道 sEMG 臂带采集肌电；Manus 动捕手套（35 个骨架关键点）提供训练期真值；部署期摘掉手套，用 HTC Vive Tracker 追踪腕部 6D 位姿（由两个追踪器的相对位移算增量）；执行端为 22 DoF 的 Sharpa Wave 手；模型部署在本地工作站。
- **标签生成：** 把人手关键点（如指尖、掌心）映射到机器人手，求解 `q* = argmin Σ‖p_h − p_r(q)‖²` 的关键点优化；用在机器人构型空间预训练的碰撞分类器检测自碰撞，一旦检测到就把姿态钳到安全流形，得到与 sEMG 同步的无碰撞 22 DoF 关节角标签。
- **EMG2Pose：** 沿用已有 EMG2Pose 架构（论文引用 NeurIPS 2024 的 emg2pose 基准）。输入 `(B, 8, T)`，两个 Conv1d 块 + 两个 TDS（Time-Depth Separable）阶段，特征 `(B, 32, T*/4)` 重采样回 `T*`；解码器 LSTM + MLP 预测关节速度 `θ̇`，再迭代 `θ_t = θ_{t−1} + θ̇_t` 重建绝对姿态，`θ_0` 为预设休息姿态；输出 `(B, 22, T*)` 动作块。论文称基于速度的目标对传感器位移与会话间噪声更稳健。
- **部署：** 在 sEMG 滑动窗口上在线推理出关节角动作块，只执行每个块的前几帧并滑动窗口；输入窗口取 400 步，论文称这是稳定性与时延的折中，偏向桌面操作的可靠性。
- **姿态精度：** 验证集平均绝对误差 MAE——抓取 0.09 rad，手内旋转 0.15 rad。
- **泛化（Table I，每类 20 次试验，SR=成功抓取并放到目标位置，DR=抓取成功后掉落的比例）：** 训练物体 SR 76.0% / DR 14.5%；未见物体 66.0% / 18.2%；新场景（随机摆放 + 杂乱遮挡背景）56.0% / 28.6%。五类物体为 Tiny、Cylinder、Sphere、Irregular、Deformable，Irregular 最差（训练物体 50.0%，新场景 25.0%）。
- **长程任务（Table II，各 20 次）：** Packaging——抓取 17/20、执行 13/20、一次性成功 12/20（60%）、允许重试 16/20（80%）；Wiping——15/20、11/20、8/20（40%）、14/20（70%）。
- **局限（论文自述）：** 新用户目前仍需个人标定；缺少力反馈，对易碎物体不友好；sEMG 信噪比不足以支撑极致触觉精度，高精度任务仍需人在回路；后续计划跨用户基础模型与触觉反馈。
- **未披露：** 训练数据时长、受试者人数、对比基线（视觉动捕 / 数据手套）的并列数字，正文均未给出。

## 对 wiki 的映射

- [DexEMG 论文实体](../../wiki/entities/paper-dexemg.md)
- [夏帕机器人（Sharpa Robotics）](../../wiki/entities/sharpa-robotics.md)

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2603.05861>
- PDF：<https://arxiv.org/pdf/2603.05861>
- Sharpa 研究博客解读页（二手）：<https://www.sharpa.com/blogs/research/breaking-the-barriers-of-teleoperation-how-dexemg-and-sharpa-wave-are-revolutionizing-dexterous-manipulation>
- 机器之心 / Sharpa 微信文章归档：[wechat_sharpa_dexterous_manipulation_2026-10-16.md](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md)
