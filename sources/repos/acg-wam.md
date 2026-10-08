# RoboOpus/ACG-WAM

> 来源归档（repo；官方训练与评测实现，核对日期：2026-10-08）

- **代码：** <https://github.com/RoboOpus/ACG-WAM>
- **项目页：** <https://RoboOpus.github.io/ACG-WAM/>
- **论文：** <https://arxiv.org/abs/2610.06965>
- **模型：** <https://huggingface.co/RoboOpus/ACG-WAM>
- **许可证：** Apache-2.0（仓库 README）
- **对 wiki 的映射：** [ACG-WAM](../../wiki/entities/paper-acg-wam-geometric-latent-prediction.md)

## 公开内容

| 能力 | 仓库入口 |
|------|----------|
| 训练 | `train/train.py`；Motus backbone + geometry JEPA |
| 教师目标缓存 | `tools/precompute_vggt_cache.py`，并提供 cache audit |
| RoboTwin 评测 | `inference/robotwin/Motus` |
| 主配方 | RoboTwin Joint ACG-JEPA，40k optimizer updates |
| checkpoint | HF 提供 RoboTwin 40k model-only 权重，约 16.09 GB |

## 复现边界

- 数据转换入口存在，但 RoboTwin 数据集不随模型仓分发；需要用户自行准备对齐的多视角视频、qpos 与文本嵌入。
- 训练依赖 Wan2.2、Qwen3-VL、Motus、VGGT 等外部模型资产；各上游资产许可需分别遵守。
- HF 下载权重适用于初始化/评测，不是含 optimizer state 的精确恢复快照。
- README 提供的是 RoboTwin evaluator policy；论文真机结果与公开视频并不构成完整公开真机软件栈。
