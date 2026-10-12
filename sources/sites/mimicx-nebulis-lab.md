# MimicX 官方项目页（NEBULIS Lab）

> 来源归档（核查日期：2026-10-12）

- **类型：** site / project page
- **主页：** <https://nebulis-lab.com/MimicX>
- **论文：** [arXiv:2610.09055](https://arxiv.org/abs/2610.09055)
- **官方代码：** <https://github.com/NEBULIS-Lab/MimicX> — [仓库归档](../repos/mimicx.md)
- **策略工件：** <https://huggingface.co/Shuaijun/MimicX-Policies>
- **数据与评测资产：** <https://huggingface.co/datasets/Shuaijun/MimicX-Assets>
- **机构：** 香港科技大学（广州）（HKUST(GZ)）
- **一句话说明：** 视频驱动人形运动跟踪研究，以策略执行反馈迭代精炼监督，并通过重复 rollout 检验候选策略。

## 项目概览

项目针对「运动参考视觉上合理，但当前控制策略在物理仿真中无法稳定执行」的情况。MimicX 将 rollout 误差反馈到训练监督：分析失败时间段和身体误差通道，提出目标函数与 reset curriculum 调整方案，继续训练后再重复执行验证。摘要报告四个核心视频任务的平均 body-tracking error 降低 25.7%、Robust Execution Horizon 提升 255.6%；详细协议与数字应以论文为准。

官方实现说明其视频到策略流程组合 GVHMR / SMPL-X 重建、GMR 重定向，以及 MuJoCo / MjLab 中的 PPO tracking；AutoRefine 负责诊断、候选生成与验证，MimicX-HLoop 支持异构执行。

## 开源核查（2026-10-12）

| 组件 | 状态 |
|------|------|
| 项目页、论文与方法说明 | **公开** |
| 官方训练 / refinement 代码 | **已开源** — 仓库含核心包、脚本、文档、配置与测试，Apache-2.0 |
| 策略和资产 | **公开** — Hugging Face 分别发布 policies 与 assets |
| 原始视频及第三方模型 / 数据 | **不随发布包提供** — GVHMR / SONIC 权重、SMPL 系列模型和第三方动作数据需依各自许可获取 |
| 结果复现 | 可复用配置、权重及记录；输入素材许可、版本和校验和需自行核对 |

## 对 wiki 的映射

- [MimicX 论文与项目实体](../../wiki/entities/paper-mimicx.md)
- [论文来源](../papers/mimicx_arxiv_2610_09055.md)
- [官方代码仓](../repos/mimicx.md)
