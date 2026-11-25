# Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness｜前沿多模态模型直接作为机器人策略的执行框架

- **类型：** paper
- **作者：** Xiao-Ming Wu、Kang Liao、Yuexiao Ma 等
- **机构：** 南洋理工大学 S-Lab、ACE Robotics、商汤研究院、厦门大学、联想机器人实验室、康涅狄格大学
- **版本：** arXiv:2611.31201v1（2026-11-23）
- **原文：** <https://arxiv.org/html/2611.31201v1> · [PDF](https://arxiv.org/pdf/2611.31201)
- **项目页：** [官方项目](../sites/embo-harness.md)
- **代码：** [官方代码](../repos/embo-harness.md)
- **核对范围：** 2026-11-25 通读 arXiv v1 正文 §1–5 与补充 §6–8，核对 Table 1–7 的开发集、全量和真机口径，及空间估计与动作位移分析；不代表本地运行源码。
- **通读：** 已完成

## 核心摘录（整理）

- 实验逐步考察动作空间与 chunk、推理强度、隐式规划、图像分辨率与多视角、短多动作工具和 VLA refinement。闭环模型依据当前观测选择空间目标；IK 或控制接口执行目标，下一次观察再修正。作者的干预实验支持空间定位精度的重要性，而非证明低层动作学习无价值。
- 论文 Table 2：RoboLab-120 SR 83.3%、Score 89.2；SenseNova-RoboRSI 为 57.1%、69.9。Table 3 真机四任务依次 5/5、5/5、2/5、3/5，共 15/20。开发子集与完整基准不可混用。
- 真机任务仅 20 次；精细定位和对齐仍易失败。模型调用时间、成本与闭环控制频率须另评。官方代码采用 NTU S-Lab License 1.0，用户提供的非商业条款仍待 LICENSE 全文核验。

- 补充 §6 的 3D 位置估计是每设置每任务一次；§7 中不同固定块长度的位移接近，但 SR 明显不同，不能用步数假定模型理解运动时长。

对 wiki 的映射：[前沿多模态模型直接作为机器人策略的执行框架](../../wiki/entities/paper-embo-harness.md)。
