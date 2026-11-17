# Reasoning with Sampling 官方 PyTorch 实现

> 来源归档（repo）

- **标题：** aakaran/reasoning-with-sampling
- **类型：** repo
- **链接：** https://github.com/aakaran/reasoning-with-sampling
- **论文：** <https://arxiv.org/abs/2510.14901>
- **项目页：** <https://aakaran.github.io/reasoning_with_sampling/>
- **入库日期：** 2026-11-17
- **沉淀到 wiki：** [独立论文节点](../../wiki/entities/paper-reasoning-with-sampling.md)
- **一句话说明：** 以 Hugging Face CausalLM 实现分块生成、低温提议与 Metropolis–Hastings 后缀重采样的实验脚本。

## 代码入口与静态调用链

- `README.md`：`conda env create -f environment.yml`、`conda activate psamp`；`sbatch llm_experiments/scripts/power_samp_math.sh` 为论文提供的 SLURM 示例。
- `llm_experiments/power_samp_math.py`：加载 `data/MATH500.json`、tokenizer、`AutoModelForCausalLM`，构造 `AutoregressiveSampler`，对每题运行基础采样、低温采样及 `mcmc_power_samp`，写 CSV。
- `llm_experiments/power_samp_utils.py`：
  - `naive_temp` 从模型生成续写并返回 proposal 归一化 log probability 与未归一化目标项；
  - `mcmc_power_samp` 默认 `block_num=16`，按 `max_new_tokens / block_num` 分块；每块重复 `mcmc_steps` 次随机起点、后缀再生成、MH 接受/拒绝；遇 EOS 截断。
  - 接受对数比率在代码中用 `target_log_prob_prop + log_prob_cur - target_log_prob_cur - log_prob_prop` 构成，说明不是只选 likelihood 最大的输出。
- `llm_experiments/eval_math.py`：读 CSV，用 `grader_utils/math_grader.py` 的 `grade_answer` 比较 `std_answer`、`naive_answer`、`mcmc_answer` 与参考答案，聚合三组准确率。
- 其他 benchmark 脚本入口 `power_samp_he.py`、`power_samp_gpqa.py`、`power_samp_alpaca.py`；未逐文件阅读，评测脚本是否在本环境可直接跑通未验证。

## 开放性与执行边界

**已开源**：官方推理及评测代码与 MATH500 输入文件；部分其他评测集需要从其官方仓库获取，原 README 已说明。仅静态代码核查，**未执行 SLURM/GPU 任务**，不主张所有依赖和脚本在任意平台开箱可用。

- **核对范围：** 通读 README、主 MATH500 采样入口、核心 power_samp_utils 实现和 MATH500 eval 脚本；检查仓库的 llm_experiments 与 scripts 目录清单。
- **通读：** 已完成
