# WorldScore 项目页（haoyi-duan.github.io）

> 来源归档（ingest）

- **标题：** WorldScore
- **类型：** site / project page
- **官方入口：** <https://haoyi-duan.github.io/WorldScore/>
- **论文：** <https://arxiv.org/abs/2504.00983>
- **代码：** <https://github.com/haoyi-duan/WorldScore>
- **数据集：** <https://huggingface.co/datasets/Howieeeee/WorldScore>
- **Leaderboard：** <https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard>
- **入库日期：** 2026-07-27
- **一句话说明：** Stanford 团队 WorldScore 官方项目页：用 next-scene + 显式相机轨迹统一评测 3D / 4D / I2V / T2V 的世界生成能力，展示 Controllability / Quality / Dynamics 指标样例与论文主表结果。
- **代码：** <https://github.com/haoyi-duan/WorldScore>（MIT，已开源）
- **数据集：** <https://huggingface.co/datasets/Howieeeee/WorldScore>

## 页面要点（2026-07-27 核查）

- **定位对比：** 现有单场景视频质量榜（如 VBench）可能给 Model A/B 相近分，但在卧室「pan left → move left → pull out」路径上，WorldScore 能区分「未生成新场景 / 未跟运镜」的失败。
- **基准对比表：** 相对 TC-Bench / EvalCrafter / FETV / VBench / WorldModelBench 等，WorldScore 宣称同时具备 Multi-Scene、Unified（3D/4D/视频）、Long Seq.、Image Cond.、Multi-Style、Camera Ctrl.、3D Consist.
- **指标分区展示：** Camera / Object Controllability、Content Alignment；3D / Photometric / Style Consistency、Subjective Quality；Motion Accuracy / Magnitude / Smoothness。
- **Evaluation Results：** 页内表格含论文批次与后续模型（如 Voyager、Wan2.1、LTX-Video）；**最新数值以 HF Leaderboard 为准**。
- **Related：** WonderJourney（CVPR 2024）、WonderWorld（CVPR 2025）。
- **BibTeX：** ICCV 2025 会议条目（`duan2025worldscore`）。

## 开源状态

**已开源** — 项目页明确链到 GitHub、HF Dataset 与 Leaderboard；无「code coming soon」占位。

## 对 wiki 的映射

- [WorldScore（论文实体）](../../wiki/entities/paper-worldscore.md)
- [worldscore 仓库归档](../repos/worldscore.md)
- [WorldScore Leaderboard（HF Space）](./worldscore-leaderboard-hf.md)
- [worldscore 论文摘录](../papers/worldscore_arxiv_2504_00983.md)
