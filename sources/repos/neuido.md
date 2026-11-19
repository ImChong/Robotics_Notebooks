# NeuIDO 官方代码仓库

> 一手代码来源归档，非本地执行结果。

- **仓库：** <https://github.com/JiajingLin/NeuIDO>
- **论文：** <https://arxiv.org/html/2609.24313v1>
- **项目页：** <https://github.com/JiajingLin/NeuIDO>（arXiv 项目链接指向该 GitHub）
- **入库日期：** 2026-11-19
- **核对范围：** README 全文与 TODO；仓库顶层目录；`operator_modules/branch_modules/branch.py`、`operator_modules/trunck_modules/trunck.py` 目录、`sim_modules/sim/`；`experiment/scripts/phase1_pretrain/train_latent_space_ours_DeepONet_seperate.py`、`phase2_vision/align_dynamics_func.py`、`phase3_finetune/finetune_model.py`、`forward/forward_zero_shot.py` 源码入口与主要训练/仿真调用路径
- **通读：** 已完成
- **映射：** [NeuIDO 独立论文实体](../../wiki/entities/paper-neuido-neural-intrinsic-dynamics-operator.md)

## 静态源码观察

- **阶段一：** `phase1_pretrain/train_latent_space_ours_DeepONet_seperate.py` 的 `F_dataset` 加载 `GtFtmp.pt`、`GtF.pt`、`GtStress.pt`、`GtC.pt`；`ElasticityTrunckNetwork`、`PlasticityTrunckNetwork` 和两组轨迹嵌入参与 AdamW 优化；保存 `elasticity` / `plasticity` 权重。
- **视觉阶段：** `phase2_vision/align_dynamics_func.py` 定义 `train_one_traj`，在可微 MPM rollout 后通过 `diff_rasterization` 与观测帧计算重建损失；源码中存在对弹性/塑性 latent 优化的路径。项目 README 尚未提供完整阶段配方，不能把这一个脚本等同于论文所述全部 Stage II。
- **少样本：** `phase3_finetune/finetune_model.py` 为弹塑性 Trunk 调用 `init_lora_layers` 和 `freeze_all_except_lora`，联合训练 LoRA 与 motion latent，并存储 LoRA state_dict。
- **前馈：** `forward/forward_zero_shot.py` 从 `VideoGuidance.get_key_frames`、`get_raft_flow` 得到视频光流，调用 `VideoBranchEncoder` 获得两路 latent，在每个 MPM 子步以弹塑性网络推进 F，按帧渲染 3DGS 并输出评测信息。仓库类名实写 `Trunck`。
- **配置依赖：** 公开脚本默认 YAML 路径以及 `infer.pretrained_branch`、`infer.pretrained_trunk` 等权重路径；这些并不能代替可获取权重与完整环境说明。

## 开源与复现判定（2026-11-19）

**部分开源。** 已可查看部分 PyTorch/Warp/3DGS 源码和实验阶段脚本，但 README 仍把 MPM 数据生成流程、依赖安装、预训练权重、完整训练/推理命令及分支架构文档列为 TODO。未核验许可文件，不推断软件许可证。未进行 CUDA/训练或 checkpoint 复现；不能把“有代码文件”写为“端到端可运行”。
