# 自动驾驶十项官方仓复核（Ten Autonomous Driving Repository Release Checks）

- **类型：** 官方仓库与发布资产复核记录
- **核对日期：** 2026-11-28（北京时间）
- **核对范围：** 十个固定提交的根 README、下表所列官方执行入口与高影响资产；论文版本供追溯，沿用各独立节点已有的全文归档。本次为静态核对，未运行训练、GPU 推理、闭环模拟或实车。
- **通读：** 已完成

| 项目详情 | 固定仓库与论文 | 版本／可复现边界 |
|---|---|---|
| [Alpamayo 1](../../wiki/entities/paper-alpamayo-r1-2511-00088.md) | [官方仓](https://github.com/NVlabs/alpamayo/tree/11a0e01c13a5622377c45ee37d653351453ec43b) · [论文 2511.00088v2](https://arxiv.org/abs/2511.00088v2) | 推理仓对应 Alpamayo 1（R1-10B），示例调用需获权访问的权重及 Physical AI AV 数据。SFT 和开环 RL 配方迁到独立 Recipes 仓；可训练不等于已下载论文的 RL 后训练权重。 |
| [OpenDriveVLA](../../wiki/entities/paper-rcl-ref-b5386c6f934f87f4cec4-opendrivevla-towards-end-to-end-autonomous-drivi.md) | [官方仓](https://github.com/DriveVLA/OpenDriveVLA/tree/10e8095bc618d508cb70cca37b6956ac4db6e9f3) · [论文 2503.23463v2](https://arxiv.org/abs/2503.23463v2) | 仓库提供推理与评测入口及 0.5B 检查点；官方发布清单的训练脚本仍未勾选。论文里的分阶段训练不能当作已经公开的复训流程。 |
| [AutoVLA](../../wiki/entities/paper-rcl-2506-13757-autovla-a-vision-language-action-model-for-end-t.md) | [官方仓](https://github.com/ucla-mobility/AutoVLA/tree/ba34eed74ce6729e7986592d0e66cbaca397b4fa) · [论文 2506.13757v3](https://arxiv.org/abs/2506.13757v3) | 仓库提供动作码本、SFT、RFT 与 NAVSIM 评测入口。官方发布的合并 LoRA checkpoint 可用 `LORA=false` 评测；快模式直接输出轨迹动作 token，慢模式先生成推理再输出 token。此仓不是 AutoVLA-D。 |
| [ReCogDrive](../../wiki/entities/paper-recogdrive-2506-08052.md) | [官方仓](https://github.com/xiaomi-research/recogdrive/tree/cdf669afb02f0e04dd9c089f4ef5cf4c8d572842) · [论文 2506.08052v2](https://arxiv.org/abs/2506.08052v2) | 官方仓库给出 2B 与 8B 的 VLM、IL、RL 检查点和训练入口；README 仍把 DriveLM、NAVSIM 2.0、DriveBench 的评测框架列为待办。论文或 README 有结果不代表同协议的评测器可重跑。 |
| [DiffusionDrive](../../wiki/entities/paper-diffusiondrive.md) | [官方仓](https://github.com/hustvl/DiffusionDrive/tree/9b52ed0ec06b073d82d6f392ab084c7b301c8681) · [论文 2411.15139v3](https://arxiv.org/abs/2411.15139v3) | 原版仓库提供 NAVSIM 和 nuScenes 权重入口。NAVSIM navtest 的 88.1 PDMS 行对应 60M、ResNet-34；nuScenes 的 ResNet-50 权重和轨迹误差表不是该分数的同一配置。不要混用原版与 DiffusionDriveV2。 |
| [DriveVLA-W0](../../wiki/entities/paper-sa-2510-12796-drivevla-w0-world-models-amplify-data-scaling-la.md) | [官方仓](https://github.com/BraveGroup/DriveVLA-W0/tree/a55766ce46ee55d9035b93816e57b8d933166482) · [论文 2510.12796v2](https://arxiv.org/abs/2510.12796v2) | 论文 §3.2 区分两种视觉监督：VQ/AR 根据过去观测和动作预测当前帧的离散视觉 token；ViT/扩散利用当前特征预测下一帧潜变量。驾驶推理通常绕过显式图像生成。公开 quick start 使用 Emu3 flow-matching 动作专家，脚本名的 87.2 与论文表中使用不同训练规模或变体的成绩不可互换。 |
| [DriveWAM](../../wiki/entities/paper-rcl-2605-28544-drivewam-video-generative-priors-enable-scalable.md) | [官方仓](https://github.com/chenshi3/DriveWAM/tree/14d16d2412dc4d70af59a921cc450299f5207f10) · [论文 2605.28544v1](https://arxiv.org/abs/2605.28544v1) | 公开训练与评测分 NAVSIM 和 PhysicalAI-AV 两条链路，模型仓分目录发布相应权重。前者用 navtest metric cache 算 PDM Score，后者有独立评测入口和导航提示。场景演化引导、联合视频–动作生成及选择性 KV 记忆不同于仅在训练期加视觉监督。 |
| [NVIDIA Cosmos](../../wiki/entities/nvidia-cosmos.md) | [官方仓](https://github.com/NVIDIA/cosmos/tree/0eab578a8935e26aa41c2b7dc9f5ab00f790a037) · [官方 Cookbook](https://github.com/NVIDIA/cosmos/blob/0eab578a8935e26aa41c2b7dc9f5ab00f790a037/COOKBOOK.md) | 根仓/配方入口与 Cosmos Framework、模型卡分开。Cosmos3-Nano-Sim-Auto 面向多视角与 LiDAR 条件的闭环驾驶仿真，Transfer-Auto 面向驾驶数据生成；它们是不同能力的示例检查点，不是可替代驾驶规划策略的同一模型。 |
| [SGDrive](../../wiki/entities/paper-sgdrive-2601-05640.md) | [官方仓](https://github.com/LogosIntelligenceLab/SGDrive/tree/d25cdce3f12e2dc8828d5db6a6add2f5dcad87f8) · [论文 2601.05640v2](https://arxiv.org/abs/2601.05640v2) | 官方 HF 文件树已列出 QA 数据与 Stage 1、IL（Stage 2）、RL（Stage 3）权重文件；README 表头的 coming soon 不应当作这些已列文件都未发布的证据。NAVSIM v2 benchmark 发布项仍未勾选。 |
| [WorldEngine](../../wiki/entities/paper-worldengine-2606-19836.md) | [官方仓](https://github.com/OpenDriveLab/WorldEngine/tree/13463cf1cbf83823578ba255f533cfce33bf3ced) · [论文 2606.19836v1](https://arxiv.org/abs/2606.19836v1) | 公开 Quick Test 用既有 VADv2 检查点在 navtest_failures 上执行闭环；官方已发布稳定检查点，并在 2026-10-27 更新 navtrain/navtest 缓存。论文完整的 Behavior World Model 集成仍是仓库未完成项，不能用 Quick Test 结果冒充论文全量后训练复现。 |

## 重点核对入口

- **Alpamayo Recipes**（固定 `622f9d23ff9e`）：[SFT](https://github.com/NVlabs/alpamayo-recipes/tree/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo1_sft)、[开环 RL](https://github.com/NVlabs/alpamayo-recipes/tree/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo1_x_rl)、[运动奖励 Python 入口](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo1_x_rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py) 与 [checkpoint 导出](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/scripts/convert_cosmos_rl_checkpoint.py)。
- **OpenDriveVLA**：[`scripts/eval_drivevla.sh`](https://github.com/DriveVLA/OpenDriveVLA/blob/10e8095bc618d508cb70cca37b6956ac4db6e9f3/scripts/eval_drivevla.sh)。
- **AutoVLA**：[`navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh`](https://github.com/ucla-mobility/AutoVLA/blob/ba34eed74ce6729e7986592d0e66cbaca397b4fa/navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh)。
- **recogdrive**：[`docs/Train_Eval.md`](https://github.com/xiaomi-research/recogdrive/blob/cdf669afb02f0e04dd9c089f4ef5cf4c8d572842/docs/Train_Eval.md)。
- **DiffusionDrive**：[`docs/install.md`](https://github.com/hustvl/DiffusionDrive/blob/9b52ed0ec06b073d82d6f392ab084c7b301c8681/docs/install.md)。
- **DriveVLA-W0**：[`Training.md`](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/Training.md)。
- **DriveVLA-W0**：[`inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh`](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh)。
- **DriveWAM**：[`src/navsim/eval.py`](https://github.com/chenshi3/DriveWAM/blob/14d16d2412dc4d70af59a921cc450299f5207f10/src/navsim/eval.py)。
- **SGDrive**：[`docs/Train_Eval.md`](https://github.com/LogosIntelligenceLab/SGDrive/blob/d25cdce3f12e2dc8828d5db6a6add2f5dcad87f8/docs/Train_Eval.md)。
- **WorldEngine**：[`scripts/closed_loop_test.sh`](https://github.com/OpenDriveLab/WorldEngine/blob/13463cf1cbf83823578ba255f533cfce33bf3ced/scripts/closed_loop_test.sh)。
- **alpamayo**：[`src/alpamayo_r1/test_inference.py`](https://github.com/NVlabs/alpamayo/blob/11a0e01c13a5622377c45ee37d653351453ec43b/src/alpamayo_r1/test_inference.py)。
- [SGDrive 发布文件清单](https://huggingface.co/SII-Whaleice/SGDrive/tree/c6379bea8aaf635883b33fcc99499b59d42aad1e)。
- [DriveWAM 发布文件清单](https://huggingface.co/chenchenshi/DriveWAM/tree/c4fb6d2c5d26ee3018a9ee1507f36fd02b85b1ce)。
- [DriveVLA-W0 发布文件清单](https://huggingface.co/liyingyan/DriveVLA-W0/tree/0bd601d6130e504d72c092d2e83e5bf05b0afa2a)。
- [WorldEngine 稳定检查点](https://huggingface.co/datasets/OpenDriveLab/WorldEngine/tree/main/data/alg_engine/ckpts/vadv2)：默认 Quick Test 路径未含 `vadv2/`，需核对本地实际布局。
