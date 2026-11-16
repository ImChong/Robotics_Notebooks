# markli1hoshipu/Plan_LLM — 官方部分实现

- **仓库：** <https://github.com/markli1hoshipu/Plan_LLM>
- **论文：** <https://arxiv.org/abs/2509.21543>（ICML 2026）
- **代码状态：** 已提供 runnable SFT / online CPO / DPO / evaluation；自动生成 PDDL domain 的核心全流程尚未集成（官方 README 明示）。许可证 Apache-2.0。
- **运行入口：** `scripts/data/generate_cot.py` → `scripts/data/generate_training_data.py` → `scripts/training/train_sft.py` → `scripts/training/run_cpo_online.sh`（调用 `train_cpo_online.py`）；评测入口 `scripts/evaluation/eval.py`。
- **复现条件：** 使用训练数据、基础模型或已公开的 Self-CriTeach/SCT 权重；不能因运行脚本可见而声称自主 domain generation 模块完整复现。
- **对应 wiki：** [Self-CriTeach](../../wiki/entities/paper-self-criteach.md)
