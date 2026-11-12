# showlab/MetaWAM — 官方代码与复现入口

- **仓库：** https://github.com/showlab/MetaWAM
- **论文：** https://arxiv.org/abs/2611.14365
- **权重：** [RoboCasa](https://huggingface.co/Ethanliang99/MetaWAM-RoboCasa-Predict2-2B)、[LIBERO](https://huggingface.co/Ethanliang99/MetaWAM-LIBERO-Predict2-2B)、[RealWorld](https://huggingface.co/Ethanliang99/MetaWAM-RealWorld-Predict2-2B)
- **核对范围：** README、ROBOCASA.md、训练/评测 shell 与 run_robocasa_eval.py、policy_text2world_model.py、policy_video2world_model.py、minimal_v1_lvg_dit.py、cosmos_utils.py 的元帧编码/掩码/动作读取相关路径；非相关文件与完整训练测试尚未通读。
- **通读：** 未完成（全仓源码；与论文直接相关的核心实现路径已逐段核验）

## 真实代码入口

- **RoboCasa 训练：** scripts/train_robocasa.sh → torchrun -m cosmos_policy.scripts.train，experiment=metawam_robocasa。
- **RoboCasa 评测：** scripts/test_robocasa.sh → python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval。
- **元帧编码：** cosmos_policy/models/policy_text2world_model.py 包含 task/current proprio/action/future proprio/value 的频域通道槽位与 IDCT 解码。
- **已发布权重：** RoboCasa checkpoint 与数据统计及 T5 embeddings；下载路径与环境依赖依官方 README。

## 核心频域实现（官方代码定位）

| 功能 | 真实实现 | 说明 |
|---|---|---|
| 固定槽位 + IDCT 注入 | [policy_text2world_model.py](https://github.com/showlab/MetaWAM/blob/main/cosmos_policy/models/policy_text2world_model.py) `_encode_frequency_meta_frame_v2` | 16 个通道：task=0；current proprio=1–3；action=4–11；future proprio=12–14；value=15，均在 `k_H=0` 系数行。row-0 IDCT 实现空间 latent 提升 |
| 读取动作 | [cosmos_utils.py](https://github.com/showlab/MetaWAM/blob/main/cosmos_policy/experiments/robot/cosmos_utils.py) `extract_action_chunk_from_meta_frame` | 对去噪输出先做正交 **2D DCT**，再读取 action 频段并 reshape；不是通过逆 DCT 解码 |
| 输入/输出元帧 + 任务路由 | [policy_video2world_model.py](https://github.com/showlab/MetaWAM/blob/main/cosmos_policy/models/policy_video2world_model.py) `_apply_input_meta_frame_conditioning`、`_build_meta_region_condition_mask`、`_apply_meta_frame_conditioning` | 条件与监督由通道掩码控制；输入元帧占用 slot 0，输出元帧复用 action slot |
| DiT 槽位掩码 | [minimal_v1_lvg_dit.py](https://github.com/showlab/MetaWAM/blob/main/cosmos_policy/_src/predict2/networks/minimal_v1_lvg_dit.py) `slot_mask_embedder` | 零初始化 residual PatchEmbed，不改变原视频主干形状，但确实引入可训练掩码输入模块 |
| 训练 | [scripts/train_robocasa.sh](https://github.com/showlab/MetaWAM/blob/main/scripts/train_robocasa.sh) → `cosmos_policy.scripts.train` | `experiment=metawam_robocasa`，默认 `NUM_GPUS=4`、`GRAD_ACCUM=4`，DCP 分片 checkpoint |
| 仿真评测 | [scripts/test_robocasa.sh](https://github.com/showlab/MetaWAM/blob/main/scripts/test_robocasa.sh) → `run_robocasa_eval.py` → `cosmos_utils.get_action` | 24 tasks × 50 trials × seeds 195/196/197；每次输出 32-step chunk、开环执行前 16 步、动作 5 步去噪 |

**维度约束：** `_get_frequency_action_slot_capacity` 以 `8 * latent_w` 为动作标量容量，超量会 assert；当前/未来本体状态槽位实现分别要求至少 9 个输入元素。适配新机器人需复核这些硬约束、数据统计量与相机配置。

**源码边界：** 已根据函数定义与官方入口核对关键数据流，尚未完整通读全仓、运行测试、训练模型或验证真实成功率和时延。不能用本项审核作为复现通过的证明。

## 复现约束

评测脚本默认 24 tasks × 50 trials × seeds 195、196、197；训练默认 4 GPU。未在本环境执行训练、评测或验证论文指标。

## 对 wiki 的映射

- [MetaWAM](../../wiki/entities/paper-metawam.md)
