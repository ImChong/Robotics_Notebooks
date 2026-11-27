# zjunlp/LabVLA — LabVLA 官方训练与部署代码仓库

- **名称：** LabVLA（`zjunlp/LabVLA`）
- **类型：** repo / vla-training / inference-deployment / data-processing
- **官方仓库：** <https://github.com/zjunlp/LabVLA>
- **对应论文：** [LabVLA（arXiv:2606.13578v2）](../papers/paper-agentic-2606-13578.md)
- **项目主页：** <https://zjunlp.github.io/LabVLA/>
- **模型权重：** <https://huggingface.co/zjunlp/LabVLA>（README 快速开始示例使用 `zjunlp/LabVLA-5B-Base`）
- **开源协议与分支：** MIT License；默认分支 `main`（快照 110 stars）
- **开源状态（2026-11-27 核对）：** **部分开源（策略训练、数据处理与部署代码已开源；数据生成引擎与数据集待发布）**。README TODO 显示 Hugging Face 模型权重、推理部署代码、训练与微调代码均已勾选完成；**RoboGenesis** 与 **LabEmbodied-Data** 仍标注为 *coming soon*。
- **核对范围：** 核对官方仓库 `README.md` 预览文本与仓库文件清单采样（`files_sample`），未本地克隆运行或执行真机/仿真测试。
- **复现边界：** **静态代码核对，未本地运行**（且环境依赖 Python 3.10、CUDA 12.6、PyTorch 2.7.1、FlashAttention 2.8.3 与 DeepSpeed ZeRO-2）。

## 已核对的目录与入口文件

1. **数据处理与校验（`files_sample` 已验证）：**
   - `data_process/__main__.py`、`data_process/cli.py`：提供 `scan` / `clean` / `stats` / `validate` / `preflight` 子命令，用于坏片段扫描、软链接清洗、归一化统计与训练前检查。
   - `data_process/cleanup/`（`scan.py`、`apply.py`）、`data_process/detectors/`（`action_dim_cap.py`、`missing_videos.py`、`stats_orphans.py`、`video_concat_hazards.py`、`video_decode.py`、`video_stubs.py`）、`data_process/validators/run.py`、`data_process/preflight.py`。
   - `data_process/labutopia_canonicalize_stats.py`、`data_process/vqa_clean.py`、`data_process/merge_v3/`（`audit_oxe_clean_v2.py`、`build_oxe_auge_clean.py`、`gripper_canonicalize.py`）。
2. **分布式训练与微调启动脚本（`files_sample` 与 `README.md` 已验证）：**
   - `configs/deepspeed_zero2.json`：DeepSpeed ZeRO-2 配置。
   - `launch/vlm_pretrain/train_vlm_pretrain.sh`：阶段 1 FAST 动作 token VLM 预训练启动脚本。
   - `launch/ki_posttrain/train_ki_posttrain.sh`：阶段 2 知识隔离（KI）流匹配后训练启动脚本。
   - `launch/finetune/train_labutopia.sh`：LabUtopia 下游任务微调启动脚本（需配置 `PretrainedCkpt`、`DataRoot`、`RepoIds`、`DatasetSchema`、`ExternalStatsPath`）。
   - README 文档指明上述 `launch/` 脚本统一通过 Accelerate + DeepSpeed ZeRO-2 调用训练入口 `scripts/train.py`，并通过 `src/schema/` 与 `schemas/` 注册 `DatasetSchema`（本次快照文件样本未完整展开 `scripts/` 与 `src/` 内部文件）。
3. **推理与仿真/真机部署（`files_sample` 已验证）：**
   - `deployment/deploy.sh`：启动推理服务的入口脚本（通过 `PRETRAINED_PATH` 指定本地权重目录）。
   - `deployment/serve_labvla.py`、`deployment/websocket_server.py`、`deployment/labsim_transforms.py`：LabVLA 推理服务、WebSocket 通信服务端与仿真坐标/状态变换模块。
4. **推理输出样例（`files_sample` 已验证）：**
   - `outputs/infer/2026.09.09/04.28.55_labvla_level3_TransportBeaker_40000_120/`：包含 `eval.log` 与 120 回合 `action_episodes/episode_*_{success,fail}.json` 评测记录。

## 对 wiki 的映射

- [LabVLA 论文实体页](../../wiki/entities/paper-agentic-2606-13578.md)
