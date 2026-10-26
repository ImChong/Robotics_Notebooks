---
type: entity
project_id: vit-h-video-to-humanoid
tags: [humanoid, video-to-humanoid, motion-generation, motion-tracking, teleoperation, diffusion, dpo, pku, unitree]
institutions: [pku]
status: complete
date: 2026-10-27
updated: 2026-10-27
arxiv: "2610.32966"
project: https://awfuact.github.io/vit-h/
code: https://github.com/mileret/ViT-H-release
related:
  - ../../wiki/methods/motion-retargeting-gmr.md
  - ../../wiki/methods/sonic-motion-tracking.md
  - ../../wiki/tasks/teleoperation.md
  - ../../wiki/tasks/loco-manipulation.md
sources:
  - ../../sources/papers/vit_h_arxiv_2610_32966.md
  - ../../sources/sites/vit-h.md
  - ../../sources/repos/mileret_vit_h_release.md
summary: "ViT-H 将单目人类视频直接映射为 Unitree G1 参考动作，再用全身跟踪结果构造偏好对、以 DPO 后训练生成器，形成视频动作生成与机器人可执行性反馈闭环。"
---

# ViT-H：带运动跟踪反馈的视频到人形控制

**一句话定义：** ViT-H 从单目人类视频直接生成 Unitree G1 的参考动作，并用机器人跟踪表现反向优化生成器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ViT-H | Video-To-Humanoid | 将视频转换为人形机器人参考动作的生成与控制框架 |
| DPO | Direct Preference Optimization | 根据 tracker 给出的动作偏好对后训练生成器 |
| AR | Autoregressive | 在线版本按时间片段生成动作并以上一段作条件 |
| G1 | Unitree G1 humanoid robot | 论文中进行实机验证的人形平台 |
| RGB | Red, Green, Blue | 在线遥操作使用的单目彩色图像输入 |

## 为什么重要

常见 video-to-humanoid 流程先从视频重建人体动作，再将动作重定向到机器人骨架，最后由 tracker 执行。ViT-H 将生成目标直接放在目标机器人的动作空间，并用下游 tracker 的执行结果优化生成器，减少分阶段优化带来的接口误差。它把“动作看起来像”与“机器人能否跟踪”放进同一训练闭环。

## 方法栈：视频条件生成与跟踪反馈

ViT-H 的运动生成器读取单目视频派生的条件，包括相机朝向、人体框、2D 关键点和图像特征，直接预测 G1 参考运动。离线版本一次生成整段动作；自回归版本利用前一动作片段，逐段生成新参考动作以支持在线输入。

\`\`\`mermaid
flowchart TD
  V["单目人类视频"] --> C["相机朝向、人体框/关键点、图像特征"]
  C --> G["扩散式 G1 动作生成器"]
  G --> X["候选机器人参考动作"]
  X --> T["可部署 tracker 与 privileged tracker 执行评估"]
  T --> P["局部/全局跟踪分数与成功信号"]
  P --> D["偏好对构造与 DPO 后训练"]
  D --> G
  G --> R["G1 全身动作跟踪"]
  V --> A["AR 生成器（逐段、条件于上一段）"]
  A --> R
\`\`\`

反馈使用两类 tracker：面向部署的 tracker 提供局部姿态与成功信号；额外观察全局信息的 privileged tracker 为全局轨迹质量提供评价。作者据此为同一视频下的候选运动排序，并通过 DPO 更新生成器。论文称偏好构造不需要人工动作标注。

## 实验与评测

- **平台与任务：** Unitree G1 实机；离线视频到人形控制和单 RGB 相机实时遥操作，演示包括行走、拳击、挥手、搬箱、擦桌、踢球、坐下/站立等。
- **实时延迟：** 论文报告自回归视频遥操作端到端延迟低于 400 ms；数字来自作者的系统评测，不代表任意硬件配置都能达到。
- **比较结果：** 作者报告 ViT-H 在动作质量与可执行性上优于论文比较的重建—重定向—跟踪基线；并报告模型容量、预训练数据及后训练数据扩大时性能改善。
- **跨 tracker：** 论文报告跟踪反馈对多个不同 tracker 的可执行性均有帮助；具体数字和实验设定见论文表格与附录。

## 与其他工作对比

| 路线 | 视频到机器人的路径 | 可执行性反馈 |
|------|--------------------|----------------|
| 经典重建—重定向—跟踪 | 视频 → 人体动作重建 → 人机重定向 → tracker | 各阶段通常分开优化 |
| ViT-H | 视频条件 → 直接生成机器人动作 → tracker | tracker rollout 构造偏好对，DPO 更新生成器 |
| SONIC / 通用 tracker | 提供多动作跟踪执行接口 | ViT-H 可将 tracker 作为生成结果评估器；二者职责不同 |

ViT-H 与 [GMR](../../wiki/methods/motion-retargeting-gmr.md) 的路线差异在于推理时是否显式输出人体动作并做运动学重定向；它也依赖可用的全身跟踪器，和 [SONIC](../../wiki/methods/sonic-motion-tracking.md) 这类通用跟踪器处在生成与执行的相邻模块。

## 结论

**一句话总判：** ViT-H 的核心贡献是把目标机器人跟踪结果反馈给视频条件动作生成器，使参考动作的可执行性进入后训练目标。

1. 直接预测机器人参考动作，省去推理链路中的人体重建与显式重定向阶段。
2. tracker 执行质量参与候选动作排序，形成生成—执行闭环；DPO 使用偏好对而非人工动作真值。
3. 在线 AR 版本面向流式视频遥操作，论文报告端到端延迟低于 400 ms。
4. 演示与论文结果依赖 G1 和对应跟踪策略，迁移到其他机器人需要重新核验动作表示与 tracker 兼容性。
5. 截至 2026-10-27，官方仓库尚无可运行代码，论文效果目前不能仅凭仓库复现。

## 工程实践与开放状态

| 项目 | 论文 / 项目页信息 |
|------|------------------|
| 输入 | 单目视频；在线演示使用 RealSense D455 RGB 流 |
| 目标输出 | Unitree G1 humanoid reference motion |
| 执行层 | 全身运动跟踪策略将参考动作转为机器人控制 |
| 在线路径 | AR 生成器按片段输出，并把参考动作送入 tracker |
| 代码状态 | 官方仓库 README 只列发布计划；截至 2026-10-27 未提供可运行训练、推理或部署代码 |

### 源码运行时序图

**不适用（截至入库日仓库没有运行代码或 README 执行步骤）**。上面的流程图表示论文系统的数据流，不是仓库可复现的运行时序。

## 局限与风险

- **系统级延迟依赖实现细节：** 论文小于 400 ms 的报告不应脱离输入处理、采样配置、硬件和缓冲策略单独外推。
- **跟踪器影响评价：** tracker 的技能范围和偏差会影响偏好排序；privileged tracker 的全局信息也不等同于部署时可观测状态。
- **代码仍待发布：** 公开仓库当前缺少可运行实现，数据、模型权重、训练超参和部署栈的可复现性需等待实际发布后检查。
- **预印本状态：** arXiv 论文标注为 under review，结论以当前版本为准。

## 关联页面

- [GMR：通用动作重定向](../../wiki/methods/motion-retargeting-gmr.md) — 视频到机器人常见显式重定向方案。
- [SONIC：规模化人形运动跟踪](../../wiki/methods/sonic-motion-tracking.md) — ViT-H 生成器下游的通用 tracker 背景。
- [遥操作（Teleoperation）](../../wiki/tasks/teleoperation.md) — 在线视频输入驱动机器人控制。
- [移动操作（Loco-Manipulation）](../../wiki/tasks/loco-manipulation.md) — 项目页演示的搬运与桌面任务类型。

## 参考来源

- [ViT-H arXiv 论文](../../sources/papers/vit_h_arxiv_2610_32966.md)
- [ViT-H 官方项目页](../../sources/sites/vit-h.md)
- [ViT-H 官方代码仓库归档](../../sources/repos/mileret_vit_h_release.md)

## 推荐继续阅读

- [ViT-H 官方项目页](https://awfuact.github.io/vit-h/) — 论文、方法示意与 G1 实机演示。
- [ViT-H 官方实现仓库](https://github.com/mileret/ViT-H-release) — README 中的代码发布计划及许可证。
