# Reactive Diffusion Policy（arXiv:2503.02881）

> 来源归档（ingest；依据论文摘要、项目页及作者公开实验表综合整理）

- **标题：** Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation
- **类型：** paper / visual-tactile imitation learning
- **arXiv：** <https://arxiv.org/abs/2503.02881>（v3，2025-04-23）
- **会议：** Robotics: Science and Systems (RSS) 2025
- **奖项：** Best Student Paper Finalist
- **作者：** Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, Cewu Lu
- **项目页：** <https://reactive-diffusion-policy.github.io/>
- **代码：** <https://github.com/xiaoxiaoxh/reactive_diffusion_policy>
- **TactAR APP：** <https://github.com/xiaoxiaoxh/TactAR_APP>
- **入库日期：** 2026-10-05

## 论文要点

1. **问题：** action chunking 有利于建模长程行为，但 chunk 执行期间难以及时响应触觉/力变化；遥操作采数也需要细粒度接触反馈。
2. **RDP：** 两级视觉–触觉模仿学习：先训练 fast asymmetric tokenizer，再训练 slow latent diffusion policy。推理时慢策略按低频视觉生成 latent action chunk；快策略基于高频 tactile/force 观测自回归修正该 chunk。
3. **TactAR：** 将传感器三维形变/力场渲染并附着在机器人末端的 AR 坐标中，支持多路 RGB 和触觉相机流。
4. **任务：** Peeling、Wiping、Bimanual Lifting；在不同触觉/力传感器与人为扰动下评估。
5. **项目页数值：** Peeling DP 0.44，RDP GelSight 0.90 / MCTac 0.88 / Force 0.95；Wiping DP 0.57，RDP GelSight 0.77 / Force 0.87；Bimanual lifting DP 0.00，RDP GelSight+MCTac 0.48 / Force 0.70。
6. **推理速度：** RTX 4090 上 DP 120 ms、RDP 慢策略 100 ms、AT 快策略 <1 ms；这是模块推理时间，不是全系统端到端延迟。
7. **用户研究：** 10 位参与者；规范化剥皮长度 0.72→0.91，稳定接触力比例 0.58→0.87。
8. **公开状态：** 项目页提供 RDP/TactAR 代码入口，并链接数据集与 checkpoint；复现环境按两个仓库 README 分别核对。

## Wiki 映射

- [RDP 深读实体](../../wiki/entities/paper-sa-2503-02881-reactive-diffusion-policy-slow-fast-visual-tacti.md)
- [ImplicitRDP 后续工作](../../wiki/entities/paper-implicitrdp-visual-force-diffusion-policy.md)
- [项目页归档](../sites/reactive-diffusion-policy-github-io.md)
- [RDP 代码归档](../repos/reactive_diffusion_policy.md)
- [TactAR APP 归档](../repos/tactar-app.md)
