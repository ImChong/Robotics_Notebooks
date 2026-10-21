# AgibotTech/WorldOracle（官方实现）

> 来源归档（repo / benchmark toolkit）

- **代码：** <https://github.com/AgibotTech/WorldOracle>
- **论文：** <https://arxiv.org/abs/2610.18826>
- **项目页：** <https://worldoracle-benchmark.github.io/>
- **数据入口：** <https://huggingface.co/WorldOracle>（项目页资源链接）
- **组织：** AgibotTech
- **许可证：** Apache-2.0
- **核查日期：** 2026-10-16
- **一句话说明：** 官方评测实现，读取预测视频、配对真实执行数据与配置，计算四项执行一致性指标并汇总分数。
- **沉淀到 wiki：** [WorldOracle 实体页](../../wiki/entities/paper-worldoracle.md)

## README 归纳（环境、数据与入口）

1. **运行环境：** Linux、Python 3.10；V-JEPA2、LPIPS、DINOv3 评测推荐使用 CUDA GPU；setup_env.sh 提供依赖安装。
2. **入口：** evaluate.py 按指标与配置运行；tools/evaluate_submission.py 支持较大提交的有界 CPU/GPU worker、安全续跑和并发物理状态评测。
3. **数据分层：** inference package 提供初始观测、动作/状态轨迹、标定和机器人运动学；ground-truth package 保存未来观测、提示、mask 与物理结果判定。模型按轨迹预测未来帧，并按 task / episode / view 组织输出。
4. **指标字段：** physical_judge、interaction_spatial_alignment、keyarea_visual_quality、multiview_consistency。分数为 0–100，越高越好；SV 为前三项均值，MV 为四项均值。
5. **VLM 评测版本：** README 记载论文分数用 GPT-5.5；live leaderboard 的基线及新提交统一用 GPT-5.6 Luna 重评，在线榜单与论文数值可能不同。
6. **多视角：** MVAC 要求同步输出 top_head、hand_left、hand_right；单视角预测不计算该项。

## 关联知识

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 动作条件世界模型与配对真机执行评测 |
| [EWMBench](../../wiki/entities/ewmbench.md) | 对照具身视频生成质量评测 |
| [WorldScore](../../wiki/entities/paper-worldscore.md) | 对照开放域世界生成评测 |

## 外部参考

- [官方 README](https://github.com/AgibotTech/WorldOracle#readme)
- [项目页](https://worldoracle-benchmark.github.io/)
- [数据入口](https://huggingface.co/WorldOracle)
