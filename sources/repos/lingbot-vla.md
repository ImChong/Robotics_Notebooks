# LingBot-VLA（1.0）

> 来源归档

- **标题：** LingBot-VLA — A Pragmatic VLA Foundation Model
- **类型：** repo
- **组织：** Robbyant（蚂蚁灵波科技 / 蚂蚁集团具身智能）
- **版本/提交：** `4eb34b76`（2026-06-12，含 2026-04-30 LeRobot v3.0 与 `torch.compile` 升级）
- **代码：** <https://github.com/robbyant/lingbot-vla>
- **项目页：** <https://technology.robbyant.com/lingbot-vla>
- **论文：** <https://arxiv.org/abs/2601.18692>（PDF：`assets/LingBot-VLA.pdf`）
- **权重：** [HF `robbyant/lingbot-vla-4b`](https://huggingface.co/robbyant/lingbot-vla-4b) / [depth 变体](https://huggingface.co/robbyant/lingbot-vla-4b-depth) / [RoboTwin 后训练权重（含/不含 depth）](https://huggingface.co/collections/robbyant/lingbot-vla) / [ModelScope 集合](https://modelscope.cn/collections/Robbyant/LingBot-VLA)
- **数据集：** [GM-100 `robbyant/gm100`](https://huggingface.co/datasets/robbyant/gm100)
- **依赖骨干：** [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal)、[LingBot-Depth](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14)
- **入库日期：** 2026-07-17
- **通读：** 已完成
- **核对范围：** `_tree.txt`（152 行）、`README.md`（326 行）、`train.sh`、`tasks/vla/train_lingbotvla.py`、`lingbotvla/models/vla/pi0/modeling_lingbot_vla.py`、`deploy/lingbot_vla_policy.py`、`configs/vla/Training_Config.md`。
- **一句话说明：** LingBot-VLA **1.0** 官方开源实现：包含 Qwen2.5-VL-3B MoT + Flow Matching（`QwenvlWithExpertModel` / `FlowMatching` / `LingbotVlaPolicy`）、LingBot-Depth 查询蒸馏头（`DepthHead` / `TaskTokenDepthHead`）、FSDP2/HSDP 高吞吐训练引擎、LeRobot v3.0 后训练配方以及支持 `torch.compile` 的 WebSocket 真机/RoboTwin 部署服务。

## 开源状态（2026-11-14 源码复核）

- **已开源（Apache-2.0）：** 完整预训练与后训练代码、**4B 预训练（w/ & w/o depth）与 RoboTwin 后训练权重**、GM-100 数据集入口、开环评测（`scripts/open_loop_eval.py`）与 WebSocket 策略服务（`deploy/lingbot_vla_policy.py`、`deploy/websocket_policy_server.py`）。

## 核心入口与模块核查

- **分布式训练入口（`train.sh`、`tasks/vla/train_lingbotvla.py`、`configs/vla/Training_Config.md`）：** `torchrun` 启动 `tasks/vla/train_lingbotvla.py`，结合 `build_foundation_model`、`build_parallelize_model`（FSDP2 / HSDP + FlexAttention + `torch.compile`）与 `VLADataset` / `VLADataCollatorWithPacking` 执行训练；`Training_Config.md` 给出真机与 RoboTwin 默认超参（`lr=5e-5`、`micro_batch_size=32`、action expert `enable_fp32=true`、8 个深度对齐 task tokens、`depth_loss_weight=0.004`）。
- **核心模型与深度蒸馏（`lingbotvla/models/vla/pi0/modeling_lingbot_vla.py`）：** `LingbotVlaPolicy` 封装 `FlowMatching`，底层为 `QwenvlWithExpertModel`（支持 AdaRMSNorm 时间条件与 KV cache 复用）；训练前向 `forward()` 对 `T=50` chunk 计算流匹配损失（`fm`/`L1_fm`），并在 `align_params` 非空时调用 `depth_emb_forward()` 与 `make_att_2d_masks_with_query()` 计算 LingBot-Depth 对齐损失；推理 `sample_actions()` 先缓存前缀图像/语言 KV，再循环调用 `predict_velocity()` 执行欧拉去噪。
- **推理服务与迁徙注意事项（`deploy/lingbot_vla_policy.py`、`README.md`）：** `LingbotVLAServer` 加载 `hf_ckpt`（`safetensors` + `config.json` + `lingbotvla_cli.yaml`）并经 `WebsocketPolicyServer` 暴露（支持 `--use_compile` 与 `--num_denoising_step`）；**注意** 2026/05/01 仓库从 LeRobot v2.1 迁往 v3.0，旧 checkpoint 需重下或删去 `config.json` 中失效的 `PI0Config` 字段。

## 对 wiki 的映射

- [LingBot-VLA](../../wiki/entities/lingbot-vla.md) — 1.0 实体页
- [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 6B / 全身动作 / MoE 后继产品
- [VLA](../../wiki/methods/vla.md) — 方法总览
