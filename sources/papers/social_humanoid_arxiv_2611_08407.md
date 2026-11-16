# Social Humanoid: Socially Responsive Human–Humanoid Interaction / 可社交响应的人–人形交互

- **类型：** 原始论文（arXiv: 2611.08407v1；2026-11-05）
- **论文：** https://arxiv.org/abs/2611.08407
- **HTML 全文：** https://arxiv.org/html/2611.08407v1
- **PDF：** https://arxiv.org/pdf/2611.08407
- **官方项目：** https://von31.github.io/social-humanoid/
- **代码开放：** Code Soon（项目页截至 2026-11-09；未见可运行仓库）
- **核查日期：** 2026-11-09
- **核对范围：** 阅读 arXiv HTML v1 的引言、方法 §3、交互数据 §4、实验 §5、结论 §6 和相关附录（重定向、数据与部署）；项目主页核查 Code Soon/Data Soon；未独立执行真机实验。
- **通读：** 已完成

## 核心机制（源自论文正文）
1. 任务：从实时人类运动线索生成机器人在双人社交互动中的响应，输入不等于事先给定的动作捕捉目标轨迹。
2. Adaptive-HHI 两阶段重定向：先对单体人形做运动迁移，再调整人类与机器人的相对互动几何，校验关节/接触/地面约束；以此构建可用的交互训练数据。
3. 模型：人体与机器人运动经共享码本 VQ-VAE 表示，Diffusion Forcing Transformer 对交互动作序列建模；部署将视觉感知、响应生成和低层全身运动跟踪串联。
4. 机器人：Unitree G1（29 DoF），使用 SONIC 全身运动跟踪控制器；交互数据约 260 小时，覆盖 CHI3D、Hi4D、Harmony4D、Inter-X、Embody3D 等来源。

## 原始实验数字及读法
- 作者报告 Adaptive-HHI 在 CHI3D 的腕部重定向误差由 1.38 cm 降至 0.50 cm，在 Harmony4D 从 2.98 cm 降至 1.91 cm；需保留原文具体任务与基准语境。
- 论文提供交互跟踪与生成效果、实时系统延迟测量；不能把视频中的交互效果当作无条件安全保证。

## 局限及复现边界
- 目前聚焦双人近身互动；缺少眼神/面部表达等模态，存在接触安全及遮挡挑战。
- 交互端到端延迟、视觉遮挡和高动态接触对部署表现影响大。

## 对 wiki 的映射
- [Social Humanoid 项目节点](../../wiki/entities/paper-social-humanoid.md)

## 参考一手资料
- [arXiv HTML v1](https://arxiv.org/html/2611.08407v1)
- [官方项目主页](https://von31.github.io/social-humanoid/)
