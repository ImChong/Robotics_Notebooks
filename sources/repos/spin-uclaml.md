# UCLA-ML SPIN 官方代码：Self-Play Fine-Tuning

> 一手源码归档

- **类型：** repo
- **机构：** 加州大学洛杉矶分校（UCLA）
- **官方 GitHub：** https://github.com/uclaml/SPIN
- **论文：** https://arxiv.org/abs/2401.01335v3
- **项目：** https://uclaml.github.io/SPIN/ （[站点归档](../sites/spin-project.md)）
- **开源状态：** 已开源；仓库提供生成、数据转换、全量微调脚本，模型与迭代数据通过 Hugging Face 发布
- **核对日期：** 2026-11-17
- **沉淀到 wiki：** [SPIN 论文实体](../../wiki/entities/paper-spin-self-play-fine-tuning.md)

## 已读代码与真实入口

| 文件或入口 | 经源码核对的行为 |
|---|---|
| `spin/generate.py` / `main` | 从输入数据集的 `real` 对提取 prompt 与参考答案；用 Hugging Face CausalLM 生成新回答；多进程 `gather_object`；写 `generated/iter*/loser_*.jsonl` |
| `spin/convert_data.py` | 合并 `loser_*.jsonl`，整理 `real/generated` 配对，输出训练和测试 Parquet |
| `scripts/finetune.sh` | 调用 `accelerate launch`，使用 `configs/deepspeed_zero3.yaml`、`spin/run_spin.py`、`configs/config.yaml` |
| `spin/run_spin.py` / `main` | `get_datasets`、`apply_chat_template`、构造 `SPINTrainer`，执行 `train()`、保存模型与 metrics |
| `spin/alignment/trainer.py` / `SPINTrainer` | `concatenated_forward` 求 real/generated 的 log prob；`get_batch_metrics` 对 policy 与冻结 opponent 求分；`spin_loss` 用 `-logsigmoid(beta * (policy_real-minus-generated - opponent_real-minus-generated))` 训练 |

## 官方最短复现路径（需依 README 调整配置）

1. 安装：Python 3.10 环境、`python -m pip install .`、`flash-attn`；配置 Hugging Face 模型权限。
2. 生成：用 `spin/generate.py` 从 SFT 数据生成模型回答，输出 `loser_{data_frac}.jsonl`。
3. 转换：执行 `spin/convert_data.py` 生成 `synthetic/train_prefs-00000-of-00001.parquet` 等文件；官方也提供预制迭代数据集，可跳过前两步。
4. 训练：`bash scripts/finetune.sh`；下一轮依次使用 `finetune_iter1.sh`、`finetune_iter2.sh`、`finetune_iter3.sh`，模型来源和数据组合由配置确定。
5. 评测：官方指向特定版本的 `lm-evaluation-harness` 与 Open LLM Leaderboard 设置。仓库 README 提醒评测版本差异会影响数值。

## 开源核查与坑点

- **2024-04-04 数据纠正：** 早期上传版本有误；应使用更正后数据。
- **SFT checkpoint 漂移：** 官方 README 固定推荐历史 revision `ac6e600eefcce74f5e8bae1035d4f66019e93190`，与已发布配对数据一致；不能直接假设最新版 SFT checkpoint 等价。
- **硬件需求：** 官方多卡全量微调示例使用 8 GPU、DeepSpeed ZeRO-3，README 注明 A100 80GB；不代表单张消费级 GPU 可直接复现。
- **验证状态：** 阅读了 README 和上述生成、转换、训练、损失代码；本次未执行训练，不声称复现了官方数字。

## 参考来源

- [SPIN 官方仓库](https://github.com/uclaml/SPIN)
- [SPIN 论文 v3](https://arxiv.org/abs/2401.01335v3)
- [SPIN 官方项目](https://uclaml.github.io/SPIN/)