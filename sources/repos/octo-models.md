# Octo（octo-models/octo）

- **URL：** <https://github.com/octo-models/octo>
- **项目页：** <https://octo-models.github.io/>
- **组织：** octo-models（UC Berkeley / Stanford / CMU / Google DeepMind）
- **版本/提交：** `241fb351`（2024-07-30）
- **许可证：** MIT
- **通读：** 已完成
- **核对范围：** `_tree.txt`、`README.md`、`octo/model/octo_model.py`、`octo/model/octo_module.py`、`octo/model/components/action_heads.py`、`scripts/train.py`、`scripts/finetune.py`、`scripts/configs/octo_pretrain_config.py`、`scripts/configs/finetune_config.py`、`examples/02_finetune_new_observation_action.py`。
- **关联论文：** [octo_arxiv_2405_12213](../papers/octo_arxiv_2405_12213.md)

## 一句话说明

基于 JAX/Flax 的开源通用机器人策略代码库：以可插拔观测/任务 tokenizer、块级因果 Transformer 主干（`OctoModule`/`OctoTransformer`）与独立读出头（`DiffusionActionHead`/`L1ActionHead`）支持从预训练到跨相机/跨动作空间微调与部署。

## 核心入口与模块核查

- **模型加载与推理（`octo/model/octo_model.py`）：** `OctoModel.load_pretrained` 从 HuggingFace（`hf://rail-berkeley/octo-small`、`octo-base`、`octo-small-1.5`、`octo-base-1.5`）加载参数、配置与数据集归一化统计量（`dataset_statistics`）；`create_tasks` 编码语言/目标图像，`sample_actions` 调用对应 `heads[head_name].predict_action` 采样动作块。
- **模块化编码与融合（`octo/model/octo_module.py`、`octo/model/components/action_heads.py`）：** `OctoModule` 将任务/观测 tokenizer 输出交由 `BlockTransformer` 执行分块因果注意力，从 `readout_action` 嵌入抽取条件向量输入 `DiffusionActionHead`（默认 20 步 cosine DDPM 残差 MLP）或 `L1ActionHead`。
- **预训练与微调入口（`scripts/train.py`、`scripts/finetune.py`、`examples/02_finetune_new_observation_action.py`）：** `scripts/train.py` 结合 `scripts/configs/octo_pretrain_config.py` 以 `make_interleaved_dataset` 混采样 25 个 OXE 数据集（`v1.5` 启用 `repeat_task_tokens=True`、GPT-3.5 `OXE_paraphrases` 复述、`use_correct_attention` 与无 dropout 扩散头）；`scripts/finetune.py` 支持通过 `merge_weights` 合并预训练权重并为新传感器/关节动作空间（`LowdimObsTokenizer`、新腕部相机、自定义 `action_dim`/`action_horizon`）初始化新 tokenizer 与读出头，RTX 4090 推理吞吐为 Base 13 it/s、Small 17 it/s。

## 交叉链接

- [Octo 论文实体](../../wiki/entities/paper-octo.md)
- [Octo 方法页](../../wiki/methods/octo-model.md)
