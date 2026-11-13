# Agentic RL Lab：官方课程入口与 GRPO 最小闭环（核验记录）

- **类型：** repo / 教程源文核对
- **项目：** <https://github.com/datawhalechina/agentic-rl-lab>
- **原始文档：** [官方 README](https://github.com/datawhalechina/agentic-rl-lab/blob/main/README.md)、[环境与基础概念](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/getting-started/readme.md)、[GRPO 详细原文](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/readme.md)、[GRPO 启动说明](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/start.md)、[同步脚本](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)
- **版本核对：** README blob `6a25b023ffd5e27b27e185351d6d54cfbb03d3e0`；GRPO 章节 blob `dbe7332ca0441394eca62dc4be331a39ea7f352e`；同步脚本 blob `775761575b0ca0fbb3ceb92652d5a0e911547a39`。
- **核对日期：** 2026-11-13
- **核对范围：** 完整阅读课程 README、起步文档、GRPO 文章与启动说明，复核同步脚本训练入口与关键模块；**仅对这些明确列出的原文通读，不声称仓库其他 16 个实验教程已逐篇读完**。
- **通读：** 已完成
- **开源状态：** 已公开 Python 实验与课程，实际训练需要 PyTRIO 远端服务、账号和算力；没有本地免费 GPU 训练的承诺。

## 课程与代表实验

README 列出 Loss Functions、GRPO、Vision GRPO、General/Medical OPD、DAPO、GSPO、OPSD、AgentOPSD、Search-R1、ReTool、ALFWorld、TEMPO、Spec-o3、Tournament-GRPO 等实验；每章是独立主题，不能从 GRPO 代表实验推断其他实验的复现结论。

## 已核对 GRPO 运行机制

1. 环境：`uv sync`、PyTRIO 登录和 SwanLab 登录；本地负责数据、奖励、循环，远程 PyTRIO 执行模型前后向、优化与采样。
2. `experiments/grpo/01-demo-sync.py` 用 `load_gsm8k_train` 加载 GSM8K；`build_prompt` 构造数学题 chat 输入。
3. 每次 step 先调用 `save_weights_and_get_sampling_client` 获取对应当前 LoRA 权重的 sampler，再用 `sample(num_samples=group_size)` 生成多条回答、保存 old-policy logprobs。
4. `extract_boxed` / `grade_answer` 比较最终 boxed 数字给 0/1 奖励，组内优势是 `reward - mean_reward`，**没有除以组内标准差**。所有优势为零的组跳过。
5. `build_grpo_datum` 把 completion tokens、old logprobs 和 advantages 右移对齐；同一套轨迹可用于 `importance_sampling`、`ppo`、自定义 `cispo`。
6. 客户端的 `forward_backward` 或 `forward_backward_custom` 发送计算，随后 `optim_step`；训练过程记录 reward、frac_degenerate、loss 等。
7. 官方 `start.md` 明确**没有独立 `eval.py`**；对比 loss 时须固定其余超参数。文中计算花费是作者实验口径，不能推广为通用固定费用。

## 对 Wiki 的映射

- [Agentic RL Lab（课程概览 + GRPO 代表实验）](../../wiki/entities/agentic-rl-lab.md)
- 与 [先前仓库归档](./agentic-rl-lab.md) 互补：本条核查深度仅覆盖上述明确列出的资料，其余实验仍需单独通读。
