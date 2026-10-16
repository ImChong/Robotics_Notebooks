# λ₀ / HumanVerse-500（arXiv:2610.18950）

> 来源归档（ingest）

- **标题：** $λ_0$: Scaling Open-World Egocentric Human Data for Humanoid Loco-Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2610.18950>
- **HTML：** <https://arxiv.org/html/2610.18950>
- **PDF：** <https://arxiv.org/pdf/2610.18950>
- **作者：** Lambda-humanoid Team
- **项目页：** <https://lambdahumanoid.github.io/>
- **代码：** <https://github.com/LambdaHumanoid/openlambda>
- **权重：** <https://huggingface.co/lambda-humanoid/lambda0-midtrain-100k>
- **提交日期：** 2026-10-13（arXiv v1，tech report）
- **一句话说明：** 以 500 小时开放世界第一视角人类移动操作数据 HumanVerse-500 训练 λ₀ 全身 humanoid VLA，并在 Unitree G1 上验证移动操作迁移。

## 核心摘录

### 1. HumanVerse-500 数据
论文将第一视角视频与同步的身体和手部运动配对，构成 500 小时开放世界人类移动操作数据集。目标是补足仅有手腕或手部轨迹时缺少的身体移动、姿态调整与物体互动协调。

### 2. λ₀ 三阶段训练
Stage I 从第一视角手物互动数据预训练；Stage II 使用机器人对齐的人类全身演示进行 mid-training；Stage III 用机器人数据进行 post-training。论文报告 λ₀ 在 SIMPLE 仿真成功率 90.0%，在所测真机移动操作任务上为 62.5%。

### 3. 开源状态（2026-10-16）
OpenLambda 是 λ₀ / HumanVerse 项目的公开代码入口，提供人类数据采集、动作对齐、仿真回放及 λ₀ 模型/训练代码。Stage II 的 100k checkpoint 已在 Hugging Face 发布。预训练与 mid-training 数据不在代码仓库中；完整 Stage III post-training 工作流、机器人数据采集与实机部署说明仍未完成，因此属于**部分开源**，不能据此视为完整可复现发布。

## 对 wiki 的映射

- [λ₀ / HumanVerse-500 实体页](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md) — 与 arXiv:2610.00438 的既有条目合并维护
- [OpenLambda 仓库归档](../repos/lambdahumanoid-openlambda.md)
- [项目页归档](../sites/lambdahumanoid-github-io.md)
