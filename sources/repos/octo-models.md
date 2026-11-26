# Octo（octo-models/octo）

- **URL：** <https://github.com/octo-models/octo>
- **项目页：** <https://octo-models.github.io/>
- **组织：** octo-models（UC Berkeley / Stanford / CMU / Google DeepMind）
- **版本/提交：** `241fb351`（2024-07-30）
- **许可证：** MIT
- **通读：** 已完成
- **核对范围：** `README.md`、`examples/01_inference_pretrained.ipynb`、`examples/03_eval_finetuned.py`、`examples/04_eval_finetuned_on_robot.py`、`octo/utils/gym_wrappers.py`、`octo/model/octo_model.py`、`octo/model/octo_module.py`、`octo/model/components/action_heads.py`、`scripts/train.py`、`scripts/finetune.py`、`scripts/configs/octo_pretrain_config.py`、`scripts/configs/finetune_config.py`、`examples/02_finetune_new_observation_action.py`。
- **关联论文：** [octo_arxiv_2405_12213](../papers/octo_arxiv_2405_12213.md)

## 一句话说明

基于 JAX/Flax 的开源通用机器人策略代码库：以可插拔观测/任务 tokenizer、块级因果 Transformer 主干（`OctoModule`/`OctoTransformer`）与独立读出头（`DiffusionActionHead`/`L1ActionHead`）支持从预训练到跨相机/跨动作空间微调与部署。

## 核心入口与模块核查

- **模型加载与推理（`octo/model/octo_model.py`）：** `OctoModel.load_pretrained` 从 HuggingFace（`hf://rail-berkeley/octo-small`、`octo-base`、`octo-small-1.5`、`octo-base-1.5`）加载参数、配置与数据集归一化统计量（`dataset_statistics`）；`create_tasks` 编码语言/目标图像，`sample_actions` 调用对应 `heads[head_name].predict_action` 采样动作块。
- **模块化编码与融合（`octo/model/octo_module.py`、`octo/model/components/action_heads.py`）：** `OctoModule` 将任务/观测 tokenizer 输出交由 `BlockTransformer` 执行分块因果注意力，从 `readout_action` 嵌入抽取条件向量输入 `DiffusionActionHead`（默认 20 步 cosine DDPM 残差 MLP）或 `L1ActionHead`。
- **预训练与微调入口（`scripts/train.py`、`scripts/finetune.py`、`examples/02_finetune_new_observation_action.py`）：** `scripts/train.py` 结合 `scripts/configs/octo_pretrain_config.py` 以 `make_interleaved_dataset` 混采样 25 个 OXE 数据集（`v1.5` 启用 `repeat_task_tokens=True`、GPT-3.5 `OXE_paraphrases` 复述、`use_correct_attention` 与无 dropout 扩散头）；`scripts/finetune.py` 支持通过 `merge_weights` 合并预训练权重并为新传感器/关节动作空间（`LowdimObsTokenizer`、新腕部相机、自定义 `action_dim`/`action_horizon`）初始化新 tokenizer 与读出头。README 所列 RTX 4090 推理吞吐不等于端到端控制频率。

## 推理与微调入口复核（2026-11-26）

- [推理 Notebook](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/examples/01_inference_pretrained.ipynb) 用 `image_primary`、时间掩码与任务字典构造输入。[sample_actions](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/model/octo_model.py#L174-L253) 默认返回归一化动作；传入正确的 `unnormalization_statistics` 时在调用内反归一化。BridgeV2 示例用 `model.dataset_statistics["bridge_dataset"]["action"]`，不应直接套到另一台本体。
- [ALOHA 仿真微调示例](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/examples/02_finetune_new_observation_action.py#L69-L125) 明确使用 `L1ActionHead`、14 维动作和 `action_horizon=50`；论文的 Berkeley Bimanual 真机任务使用 64 步预测块，重规划前执行 12 步。两者不可互当复现配置。默认 [WidowX 真机示例](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/examples/04_eval_finetuned_on_robot.py#L100-L130) 使用时间集成封装，滚动时域封装是替代选项。
- 复核范围是代码路径和示例参数，不含下载权重后跑 GPU 或连接真机。

## 交叉链接

- [Octo 论文实体](../../wiki/entities/paper-octo.md)
- [Octo 方法页](../../wiki/entities/paper-octo.md)
