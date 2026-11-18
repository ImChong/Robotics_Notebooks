# LeapLabTHU/limit-of-RLVR：数学与编程 pass@k 评测代码

> 来源归档（repo）

- **类型：** repo
- **代码：** https://github.com/LeapLabTHU/limit-of-RLVR
- **论文：** https://arxiv.org/abs/2504.13837
- **项目页：** https://limit-of-rlvr.github.io/
- **入库日期：** 2026-11-18
- **沉淀到 wiki：** [Limit of RLVR 独立节点](../../wiki/entities/paper-limit-of-rlvr.md)

## 官方入口静态核查

- **数学入口：** math/README.md 给出 conda Python 3.10、torch 2.4、flash-attn 安装及 bash eval_math_nodes.sh 评测命令；参数包括 --init_model、--benchmarks、--temperature 0.6、--top_p 0.95、--n_sampling 32、--seed。
- **数学调用链：** math/eval_math_nodes.sh 进入 examples/math_eval，按 checkpoint 分组并调用 sh/convert_and_evaluate_gpu_nodes.sh；该脚本按 GPU 组调用 sh/eval.sh；后者执行 math_eval.py --use_vllm 产生采样/判题输出；sh/collect_results.py 汇总。math/pass@k.py 从 jsonl score 序列使用 1−C(n−c,k)/C(n,k) 计算 pass@k 均值。
- **编程入口：** code/DeepCoder/README.md：先获取任务数据，再调用 scripts/eval/eval_model.sh --model --datasets --output-dir --n --tp --max-length --seed；脚本运行 python3 -m verl.trainer.main_generation_vllm2，写 parquet，并在 README 指向 results.json 汇总。
- **开源界限：** 仓库已开放数学和 DeepCoder 评测代码；上述入口用于评测，并不等同作者发布论文全部训练配置、检查点和视觉实验的完整复现流水线。

## 运行环境风险

- math/eval_math_nodes.sh 仍含 init_model_path 空值和 WANDB_API_KEY=TO_BE_FILLED 占位，使用前需要配置本地模型路径、环境与密钥；不可直接照抄为开箱即跑。
- code/DeepCoder/scripts/eval/eval_model.sh 含硬编码 HOME（示例机器绝对路径），且调用包含 8 GPU 配置的 vLLM/verl 环境；需要自行调整 GPU 和数据目录。
- **执行状态：** 静态检查官方 README/入口/分析脚本；未安装依赖、未运行 GPU 采样、未独立复核论文数值。
- **核对范围：** 阅读仓库 README、math/README.md、math/eval_math_nodes.sh、math/examples/math_eval/sh/convert_and_evaluate_gpu_nodes.sh、sh/eval.sh、math/pass@k.py、code/DeepCoder/README.md 与 scripts/eval/eval_model.sh，并核对其参数与数据流。
- **通读：** 已完成
