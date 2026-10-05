# ImplicitRDP（arXiv:2512.10946）

> 来源归档（ingest；依据论文摘要、HTML 全文与作者项目页）

- **标题：** ImplicitRDP: An End-to-End Visual-Force Diffusion Policy With Structural Slow-Fast Learning
- **类型：** paper / end-to-end visual-force diffusion policy
- **arXiv：** <https://arxiv.org/abs/2512.10946>（v2，2026-07-21）
- **期刊：** IEEE Robotics and Automation Letters (RA-L), 2026, Vol. 11, No. 8, pp. 10010–10017
- **DOI：** <https://doi.org/10.1109/LRA.2026.3710031>
- **作者：** Wendi Chen, Han Xue, Yi Wang, Fangyuan Zhou, Jun Lv, Yang Jin, Shirun Tang, Chuan Wen, Cewu Lu
- **项目页：** <https://implicit-rdp.github.io/>
- **代码：** <https://github.com/Chen-Wendi/ImplicitRDP>
- **入库日期：** 2026-10-05

## 论文要点

1. **问题：** 视觉提供稠密空间但低频的全局上下文，力传感反映快速局部接触状态；强行同频融合会损失时间角色。RDP 通过显式慢快两级解耦，但快层会受 latent 信息瓶颈和慢层错误约束。
2. **Structural Slow-Fast Learning（SSL）：** 用因果注意力将异步视觉 token 与动作率力 token 放进统一模型，保持 chunk 级动作连贯，并能在 chunk 内以新力反馈闭环更新。
3. **Consistent inference：** 复用慢上下文与 chunk 采样状态，在快环内接收新力观测并输出当前动作，旨在支持动作率反应而保持扩散采样时序。
4. **Virtual-target-based Representation Regularization（VRR）：** 按 compliance 关系从实测位姿与外力构造虚拟目标，将力相关辅助表示对齐动作空间，减少 modality collapse。
5. **准静态示意关系：** (x_v = x_{real} + K^{-1} f_{ext})。符号、坐标与 K 的定义以论文公式及 controller 章节为准。
6. **评测：** Box flipping 与 switch toggling 各 20 次：DP 0/20、8/20；RDP 16/20、10/20；ImplicitRDP 18/20、18/20。消融：完整方法 18/20、18/20；去掉 SSL 与 VRR 为 6/20、5/20；去掉 SSL 为 4/20、15/20。
7. **边界：** 与 RDP 前作任务/指标不同，不能和前作剥皮/擦拭分数直接排名；小规模试验仅支持论文给定配置下的对照结论。

## Wiki 映射

- [ImplicitRDP 实体页](../../wiki/entities/paper-implicitrdp-visual-force-diffusion-policy.md)
- [RDP 前作实体页](../../wiki/entities/paper-sa-2503-02881-reactive-diffusion-policy-slow-fast-visual-tacti.md)
- [项目页归档](../sites/implicit-rdp-github-io.md)
- [代码仓归档](../repos/implicitrdp.md)
