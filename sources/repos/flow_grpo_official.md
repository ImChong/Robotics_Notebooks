# Flow-GRPO 官方源码走读（yifan123/flow_grpo）

- **官方仓库：** https://github.com/yifan123/flow_grpo （README 声明 NeurIPS 2025 official implementation，MIT）
- **论文：** https://arxiv.org/abs/2505.05470
- **项目主页：** https://gongyeliu.github.io/Flow-GRPO/
- **核对时间：** 2026-11-10
- **核对范围：** 已阅读 README Quick Started 与 Changelog、`scripts/single_node/grpo.sh`、`scripts/train_sd3.py`（`TextPromptDataset` / `GenevalPromptDataset`、`main` 采样→异步奖励→标准化优势→GRPO 更新）、`flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`（轨迹和逐步 log-prob）、`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`（随机 transition / 似然）、`flow_grpo/rewards.py` 的奖励封装，以及 `config/grpo.py` SD3 OCR / GenEval 配置入口。没有执行训练、显存测试或重现定量结果。
- **通读：** 已完成

## 真实代码入口与调用结构

| 阶段 | 源码与核验发现 |
|---|---|
| 启动 | `bash scripts/single_node/grpo.sh` → `accelerate launch ... scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu`；README 提供单机 `grpo.sh`、多机 `multi_node/sd3.sh` |
| 数据 | `TextPromptDataset` / `GenevalPromptDataset` 从文本或 metadata JSONL 读取 prompts，`DistributedKRepeatSampler` 分组重复采样 |
| 文本条件 | `compute_text_embeddings` 使用 SD3 的 text encoders / tokenizers |
| rollout | `pipeline_with_logprob` 调用 SD3 Transformer 并迭代 scheduler，底层 `sde_step_with_logprob` 返回每步 latent 与 log_prob |
| 奖励 | `executor.submit(reward_fn, images, prompts, metadata)` 在样本生成后异步打分；`flow_grpo.rewards.multi_score` 按配置选择奖励 |
| 优势 | `PerPromptStatTracker.update` / group-wise reward normalization，随后为每步 broadcast advantage，并跳过纯零优势样本 |
| 训练 | `compute_log_prob` 重算当前 policy 的逐步 log_prob，`ratio=exp(new-old)`，PPO-style clipped objective；`beta>0` 时通过 `disable_adapter()` 得到 reference mean 并计算 KL |
| 优化与评测 | `Accelerator` 梯度累积 / 更新，`eval` 定期基于评估 prompts 生成结果并打分，`save_ckpt` 保存 |

## 源码审阅边界

- 本仓 **已经含可运行的训练脚本**，可以据此绘制源码 sequenceDiagram；但这里仅作**静态源码走读**，未实测命令成功运行。
- `config/grpo.py` 提供多模型与奖励任务配置；论文 `G=24` 与 24×A800 是原论文实验配置，README 的单 GPU 启动脚本只是复现入口，不能据此改写论文训练资源。
- 官方 README 自 2025 年后又加入 `Flow-GRPO-Fast`、`GRPO-Guard`、`CPS` 与更多后端，这些不是原始 v5 方法的必要组成。
- Diffusers patch 中 `sde_type` 有后续扩展的 `cps` 分支；讲解原论文应使用 `sde` 分支。

## 对 wiki 的映射

- [Flow-GRPO 主实体](../../wiki/entities/paper-flow-grpo.md)
- [论文通读](../papers/flow_grpo_arxiv_2505_05470_reading.md)
- [官网核验](../sites/flow_grpo_project.md)
