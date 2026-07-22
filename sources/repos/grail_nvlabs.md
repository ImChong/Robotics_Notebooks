# GRAIL（NVlabs/GRAIL）

> 来源归档（ingest · 官方代码仓库）

- **标题：** GRAIL — Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
- **类型：** repo
- **URL：** <https://github.com/NVlabs/GRAIL>
- **机构：** NVIDIA
- **入库日期：** 2026-06-30
- **复核日期：** 2026-07-22
- **许可证：** NVIDIA License（README 说明非商业使用限制；第三方组件另遵许可）
- **文档：** <https://nvlabs.github.io/GRAIL/>
- **数据集：** <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
- **一句话说明：** NVIDIA 官方 GRAIL 数字数据生成管线代码入口：从 3D 资产与视频先验合成人形 loco-manipulation 示范，支撑 4D HOI 重建、G1 retargeting、task-general tracking、数据导出与 sim-to-real 视觉策略训练。

## 核心摘录（策展，非全文）

- **定位：** 与 [arXiv:2606.05160](../papers/grail_arxiv_2606_05160.md) 及 [项目页](https://research.nvidia.com/labs/dair/grail/) 配套的官方实现仓库。
- **Quick Start：** README 使用 `docker.io/nvgrail/grail:latest`；容器内运行 `scripts/setup/install_env_docker.sh`、`download_checkpoints.sh`、`download_comasset.sh`。
- **Pipeline 入口：** README 明确要求使用 package entrypoints：`python -m grail.pipelines.gen_terrain`、`gen_3d_assets`、`gen_2dhoi`、`recon_4dhoi`。
- **凭据边界：** `.env` 可包含 `OPENAI_API_KEY`、`KLING_*`、`HF_TOKEN`，说明完整 2D HOI 生成依赖外部 API / HF 权限。
- **已发布：** task-general tracking policy checkpoints（README TODO 已勾选）。
- **待发布：** quick-start demo script、GRAIL manipulation dataset（README TODO 未勾选）。

## 运行入口摘要

| 阶段 | README 入口 |
|------|-------------|
| 环境 | `docker pull docker.io/nvgrail/grail:latest`；`docker run --gpus all ...` |
| 安装 | `bash scripts/setup/install_env_docker.sh` |
| 权重 | `bash scripts/setup/download_checkpoints.sh` |
| 示例资产 | `bash scripts/setup/download_comasset.sh --category cordless_drill` |
| 3D terrain | `python -m grail.pipelines.gen_terrain --type stairs --num 50 --output_dir data/syn_stairs` |
| 3D assets | `conda run -n hunyuan python -m grail.pipelines.gen_3d_assets -i configs/gen_3d/example_objects.yaml -o data/gen_example` |
| 2D HOI | `python -m grail.pipelines.gen_2dhoi --dataset ComAsset --category cordless_drill --results_dir results --video_model_api kling-ai` |
| 4D HOI | `python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill --results_dir results` |

## 对 wiki 的映射

- [paper-grail](../../wiki/entities/paper-grail.md)
- [grail-locomanipulation-dataset](../../wiki/entities/grail-locomanipulation-dataset.md)

## 参考来源（原始）

- GitHub：<https://github.com/NVlabs/GRAIL>
- 项目页：<https://research.nvidia.com/labs/dair/grail/>
- README（2026-07-22 抓取）：<https://raw.githubusercontent.com/NVlabs/GRAIL/main/README.md>
