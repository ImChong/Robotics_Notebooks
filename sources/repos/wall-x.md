# WALL-X（X-Square-Robot/wall-x）

> 来源归档（国内具身开源全景入库；2026-10-09 按 README 原文复核更新）

- **标题：** Wall-X — WALL 系列开源具身基础模型的训练与推理代码
- **类型：** repo
- **机构：** 自变量机器人（X Square Robot）
- **链接：** <https://github.com/X-Square-Robot/wall-x>（旧记录写作 `WALL-X`；GitHub 仓库名大小写不敏感）
- **许可：** Apache-2.0（仓库根目录 `LICENSE`）
- **版本：** `setup.py` 中 `wall_x` **1.1.0**
- **配套论文：** WALL-OSS [arXiv:2509.11766](https://arxiv.org/abs/2509.11766)；Wall-OSS-0.5 [arXiv:2605.30877](https://arxiv.org/abs/2605.30877)
- **项目页：** <https://x2robot.com/en/research/68bc2cde8497d7f238dde690>（WALL-OSS）；<https://x2robot.com/oss>（Wall-OSS-0.5）
- **分类：** VLA / 具身基础模型
- **首次入库：** 2026-09-06（[国内具身开源全景](../blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)）
- **复核日期：** 2026-10-09（来源：raw README、`workspace/README.md`、`scripts/README.md`、`setup.py`、`LICENSE`；github.com 的 HTML 页面与 API 在本环境返回 403，因此 release/commit 列表未核对）

## README 要点（2026-10-09）

- **定位：** "training and inference code for the WALL series open-source embodied foundation models"，内容包括 LeRobot 数据准备、模型配置、**flow-matching 与 FAST 两类动作分支**、公开的服务与评测工具，以及安装时编译的 CUDA 算子源码（`wall_x/model/core/ops/csrc/`）
- **News：** 2026-06 Wall-X 1.1.0 更新了 Wall-OSS-0.5 的训练推理栈（公开 serving/eval runtime、DMuon 训练支持、安装时 CUDA 算子编译）；2026-05 WALL-WM（arXiv:2606.01955）；2026-05 Wall-OSS-0.5（arXiv:2605.30877v2）；2025-09 WALL-OSS（arXiv:2509.11766）
- **Models：** HF `wall-oss-0.5` / `wall-oss-flow-0.1` / `wall-oss-flow` / `wall-oss-fast`
- **环境：** Python 3.10、CUDA 12.x、Ubuntu 22.04；`flash-attn==2.8.3`；`pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"`；LeRobot 用 `--no-deps` 安装；`MAX_JOBS=8 pip install --no-build-isolation -e .`
- **版本切换：** `workspace/README.md` 写明"本次开源面向 Wall-OSS-0.5"；若用 WALL-OSS-FLOW / FAST，需要 `git checkout 97406f2ab5de414c79b091873f946c112d105c72`

## 运行入口

| 环节 | 入口 |
|------|------|
| 微调 | `python -m wall_x.trainer.fsdp_trainer.train_fsdp --config <yml>`；多卡 `torchrun --nproc_per_node=4 wall_x/trainer/fsdp_trainer/train_fsdp.py`；单卡至少需要 **48 GB** 显存 |
| 配置模板 | `workspace/example/qwen2_5_lerobot_template.yml`、`libero.yml`、`maniparena_example.yml`（双臂、448px、三相机）、`arrange_3_flowers_wrc_red.yml` |
| 归一化 | `scripts/compute_norm_stats.py`（LeRobot v3 parquet） |
| 冒烟推理 | `scripts/fake_inference.py --checkpoint-path`（`Qwen2_5_VLMoEForAction.from_pretrained()`，bf16） |
| 仿真评测 | `scripts/run_libero.sh` → `infer_libero.py`（4 个 LIBERO suite；LIBERO 的 7 维动作要 pad 到 26 维） |
| 服务 | `scripts/run_serving.sh --checkpoint-path --train-config-path --port 32195`（WebSocket；默认返回原始 action chunk，加 `--serialize-actions` 返回机器人序列化动作） |
| 开环评测 | `scripts/draw_openloop_plot.py --uri ws://... --dataset-root --train-config` |
| 部署 | `workspace/rtx5090/`（RTX 5090 安装与服务脚本） |
| 工具 | `merge_sharded_weights.py`（合并 FSDP 分片）；`merge_tokenizer.py`（把 FAST token 并入 Qwen2.5-VL processor） |

## 开源状态

- **已开源**：训练、微调、LIBERO 评测和 WebSocket 推理代码（Apache-2.0）；四个 HF 权重
- **未随仓发布**：预训练语料（自采数据、具身 VQA、Wall-OSS-0.5 的 12M bridge 样本）、真机评测套件；README 只给出微调与评测流程，没有预训练复现说明（仓库目录没有逐一列出核对）

## 对 wiki 的映射

- [wiki/entities/cn-os-wall-x.md](../../wiki/entities/cn-os-wall-x.md)（WALL-OSS 与 WALL-X 主节点）
- [wiki/entities/paper-wall-oss-0-5.md](../../wiki/entities/paper-wall-oss-0-5.md)（Wall-OSS-0.5 技术报告）
- 官方页归档：[sources/sites/x2robot-wall-oss.md](../sites/x2robot-wall-oss.md)
