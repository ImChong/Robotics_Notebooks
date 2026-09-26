# Abhiram824/egoexo4d_human_meshes

> 来源归档

- **标题：** egoexo4d_human_meshes（Ego-Exo4D-HM 官方代码）
- **类型：** repo
- **组织 / 作者：** Abhiram Maddukuri、Georgios Pavlakos（UT Austin）
- **代码：** <https://github.com/Abhiram824/egoexo4d_human_meshes>
- **论文：** <https://arxiv.org/abs/2609.30187>
- **项目页：** <https://abhiram824.github.io/egoexo4d_human_meshes/>
- **预计算数据：** <https://huggingface.co/datasets/Ego-Exo4D-HM/npz-datasets>
- **许可：** MIT（项目页 Overview；基于 [SLAHMR](https://vye16.github.io/slahmr/)）
- **入库日期：** 2026-09-26
- **一句话说明：** Ego-Exo4D 多视角 SMPL-H 重建管线：去畸变 → 相机 → ViTPose/HaMeR → 三角化 → SLAHMR 优化；`run_pipeline.py` 批处理与 HF npz 发布。

## 入口速查（对齐项目页 Installation / Pipeline · 2026-09-26）

| 路径 / 命令 | 作用 |
|-------------|------|
| `export EGOEXO4D_DATASET=...` | 指向已下载 Ego-Exo4D（`slahmr/macros.py` import 时校验） |
| `bash install_conda.sh` / `install_pip.sh` | 环境（含 detectron2、neural-renderer、lietorch 等 CUDA 扩展） |
| `scripts/run_pipeline.py --video <take> --device_num 0` | 单 take 五阶段生产入口 |
| `scripts/run_pipeline.py --jobs_file jobs.jsonl` | 多 GPU 文件锁批处理 |
| `hf download Ego-Exo4D-HM/npz-datasets --repo-type dataset` | 跳过自跑，拉预计算 npz |
| `scripts/run_mesh_vis_hands_egoexo.py --npz_path ...` | npz → 各 exo 视角 mesh 渲染视频 |

## 项目页 / 源码开放核查（步骤 2.5）

- **状态：已开源** — 项目页提供完整安装、模型下载与 pipeline 文档；HF 数据集已发布。
- **边界：** 非独立视频语料；必须先有 Ego-Exo4D 原始 take 与标定；Ubuntu 22.04 + CUDA devel 为文档测试环境。

## 与本仓库知识的关系

- 论文归档：[`sources/papers/egoexo4d_hm_arxiv_2609_30187.md`](../papers/egoexo4d_hm_arxiv_2609_30187.md)
- 项目页：[`sources/sites/egoexo4d-hm-abhiram824.md`](../sites/egoexo4d-hm-abhiram824.md)
- wiki：[`wiki/entities/paper-egoexo4d-hm.md`](../../wiki/entities/paper-egoexo4d-hm.md)
- 方法基底：SLAHMR（Ye et al., CVPR 2023）；手部 HaMeR（Pavlakos et al., 2024）
