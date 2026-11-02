# CaRE 官方代码归档

- **仓库：** https://github.com/LMMMEng/CaRE （Apache-2.0）
- **论文：** https://arxiv.org/abs/2602.03473v2
- **数据：** https://huggingface.co/datasets/LMMM2025/OmniBenchmark-1K
- **实体：** [CaRE](../../wiki/entities/paper-care-bi-level-routing.md)
- **核对：** 2026-11-02，README、训练脚本、`main.py`、`trainer.py`、`models/care.py` 和 `backbone/vit_brmoe.py`。

README 提供 PyTorch 训练实现与数据下载入口。301 任务示例为 `bash scripts/omnibenchmark1k/run_ob1k_301task_b100inc3.sh`，该脚本将对应 JSON 配置传给 `main.py`。环境参考 Python 3.10、CUDA 12.1、PyTorch 2.2.2、timm 0.6.13；数据目录按 README 配置。未实际下载数据或训练。

`trainer.py` 创建 DataManager 与 Learner，逐任务调用 `incremental_train`、`eval_task`、`after_task` 并保存权重。`models/care.py` 建立训练/测试数据加载器，扩展分类头及 adapter，训练 BR-MoE；增量任务还执行特征统计及分类器对齐。长序列配置采用单次推理路径，切勿将短序列的迭代推理成本直接套用到长序列。
