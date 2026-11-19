# RoboticsDiffusionTransformer（thu-ml/RoboticsDiffusionTransformer）

- **URL：** <https://github.com/thu-ml/RoboticsDiffusionTransformer>
- **项目页：** <https://rdt-robotics.github.io/rdt-robotics/>
- **论文：** [rdt_1b_arxiv_2410_07864.md](../papers/rdt_1b_arxiv_2410_07864.md)
- **License：** MIT
- **入库日期：** 2026-09-28
- **最近复核：** 2026-11-19
- **核对范围：** `README.md`、`main.py`、`train/train.py`、`models/rdt_runner.py`、`models/rdt/model.py`、`models/rdt/blocks.py`、`configs/base.yaml`、`scripts/agilex_inference.py`、`scripts/agilex_model.py`
- **通读：** 已完成

## 一句话说明

RDT-1B 官方 PyTorch + DeepSpeed 实现：包含 1.2B/170M 去噪主干 `RDT`、统一 128D 物理空间适配器 `RDTRunner`、TFRecord 生产者-消费者预训练与 HDF5 微调管线、ManiSkill 仿真评测及 Mobile ALOHA 真机部署栈。

## 关键入口与模块（源码核对）

| 路径 / 符号 | 用途 | 关键实现细节 |
|-------------|------|--------------|
| `main.py` (`parse_args`) | CLI 总入口 | 解析 `--pretrained_model_name_or_path`、`--dataset_type`（`pretrain`/`finetune`）、`--load_from_hdf5`、`--precomp_lang_embed`、`--deepspeed` 等参数并调用 `train()` |
| `train/train.py` (`train`) | 训练循环 | 使用 `Accelerator` + DeepSpeed ZeRO-2 加载 `T5Embedder`（`t5-v1_1-xxl`）、`SiglipVisionTower`（`siglip-so400m-patch14-384`）与 `VLAConsumerDataset`/`HDF5VLADataset`，构建 `RDTRunner` 执行去噪训练与周期性 `overall_avg_sample_mse` 采样验证 |
| `models/rdt_runner.py` (`RDTRunner`) | 扩散策略包装器 | 持有 `DDPMScheduler`（1000 步 `squaredcos_cap_v2`）与 `DPMSolverMultistepScheduler`（默认推理 5 步）；`adapt_conditions` 对齐语言/图像特征，`compute_loss` 训练去噪 MSE，`conditional_sample` / `predict_action` 输出 64 步动作块 |
| `models/rdt/model.py` & `blocks.py` (`RDT`, `RDTBlock`) | 1.2B DiT 主干 | `TimestepEmbedder` 编码步数与频率；多视角时序位置编码 `get_multimodal_cond_pos_embed`；28 层 `RDTBlock` 使用 `RmsNorm`、`CrossAttention`（带 `qk_norm`）交替注入 `[lang_c, img_c]`，末尾接非线性 `FinalLayer`（MLP 解码器） |
| `configs/base.yaml` | 模型与数据配置 | 定义 `hidden_size: 2048`、`depth: 28`、`num_heads: 32`、`pred_horizon: 64`、`img_history_size: 2`、`state_token_dim: 128` 等默认超参 |
| `scripts/agilex_inference.py` & `agilex_model.py` (`RoboticDiffusionTransformerModel`) | 真机部署 | 封装 ROS 相机/关节订阅与 25 Hz（支持 `--use_actions_interpolation` 插值）闭环，`step()` 调用 `_format_joint_to_state`（128D + mask）与 `_unformat_action_to_joint` 完成真机推理 |
| `eval_sim/eval_rdt_maniskill.py` | 仿真评测（README） | 报告 5 项 ManiSkill 任务平均 **53.6±0.52%**（vs Diffusion Policy 30.2%、OpenVLA 4.8%、Octo 0.0%）；另附 RoboTwin 2.0 榜单对齐链接 |

## 交叉链接

- [paper-rdt-1b](../../wiki/entities/paper-rdt-1b.md)
