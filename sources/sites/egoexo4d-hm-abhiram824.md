# Ego-Exo4D-HM 项目页

> 来源归档（ingest）

- **标题：** Ego-Exo4D Human Meshes Dataset（Ego-Exo4D-HM）
- **类型：** site（作者 GitHub Pages 文档站）
- **发布方：** Abhiram Maddukuri、Georgios Pavlakos（UT Austin）
- **原始链接：** <https://abhiram824.github.io/egoexo4d_human_meshes/>
- **论文：** <https://arxiv.org/abs/2609.30187>
- **代码：** <https://github.com/Abhiram824/egoexo4d_human_meshes>
- **数据集：** Hugging Face `Ego-Exo4D-HM/npz-datasets`（dataset）
- **入库日期：** 2026-09-26
- **一句话说明：** 官方文档站：数据集概览（~523 h 多视角视频 / 8 类活动）、Installation（conda/pip + Ego-Exo4D 前置）、HF 预计算 npz 下载、五阶段 `run_pipeline.py` 与渲染脚本、npz 字段说明；MIT，基于 SLAHMR。

## 项目页 / 源码开放核查（步骤 2.5 · 2026-09-26）

| 核查项 | 结论 |
|--------|------|
| **导航 GitHub** | 链至 `github.com/Abhiram824/egoexo4d_human_meshes` |
| **Installation** | `git clone --recursive`、子模块（含 ViTPose fork）、`install_conda.sh` / `install_pip.sh`、`download_models.sh`；要求 CUDA devel + `EGOEXO4D_DATASET` |
| **Download** | 原始 [Ego-Exo4D](https://ego-exo4d-data.org/) 必下；预计算 **2649 takes** 在 HF `Ego-Exo4D-HM/npz-datasets`（~48GB） |
| **Pipeline** | 生产入口 `scripts/run_pipeline.py`；支持 jobs 文件批处理、`--resume_stage` |
| **开放程度** | **已开源**：代码文档与 HF 数据均已发布；复现仍依赖 Ego-Exo4D 访问与 GPU 环境 |

## 摘录要点（与论文分工）

- **对外叙事：** 为 Ego-Exo4D 多视角采集补上 **SMPL-H 4D mesh / 关节** 层，使「画面 ↔ 身体运动」可对齐使用。
- **工程入口：** 可只下 HF npz + 原始 take 视频做渲染/评测，也可自跑完整 SLAHMR 改造管线。

## 对 wiki 的映射

- [Ego-Exo4D-HM（论文实体）](../../wiki/entities/paper-egoexo4d-hm.md)
- 姊妹归档：[论文摘录](../papers/egoexo4d_hm_arxiv_2609_30187.md)、[代码仓](../repos/egoexo4d-human-meshes.md)
