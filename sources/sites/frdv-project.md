# FRDV 官方项目主页（DARoS Lab）

> 来源归档（project site / official research page）

- **名称：** FRDV — Fast Retargeting with Dynamics Validation
- **项目主页：** <https://daroslab.github.io/frdv/>
- **论文：** <https://arxiv.org/abs/2610.28470>
- **代码：** <https://github.com/DARoSLab/FRDV>
- **演示：** 项目页视频与仓库 `assets/demo/`
- **核对日期：** 2026-11-12
- **核对范围：** 项目介绍、论文入口、代码入口、演示与发布情况；结合代码仓库 README 核对公开边界。
- **通读：** 已完成

## 项目页公开内容

- FRDV 将大规模人体动作数据映射为目标人形机器人参考轨迹。论文把运动学 SQP/RTI 与可选 CI-iLQR 动力学验证分开。
- 论文实验包含运动学基准、下游动作跟踪策略及 Unitree G1 Sim2Real 展示。
- 项目页链接的官方代码为 **DARoSLab/FRDV**，不是只有论文或演示视频。
- **部分开源结论：** 截至本次核对，公开仓库的 2.0 README 明确**只发布运动学重定向代码，不含论文中的动力学验证实现**。不要因为项目名含 Dynamics Validation 就误以为 CLI 已实现该阶段。

## 关联归档与 wiki

- [论文全文归档](../papers/fast-motion-retargeting-pipeline-humanoid_arxiv_2610_28470.md)
- [官方代码归档（FRDV 2.0）](../repos/frdv.md)
- [FRDV 唯一知识节点](../../wiki/entities/paper-fast-motion-retargeting-pipeline-humanoid.md)
