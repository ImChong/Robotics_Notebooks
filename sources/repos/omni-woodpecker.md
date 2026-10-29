# Omni-Woodpecker 官方代码

> 来源归档（repo）

- **代码：** <https://github.com/ZiiJianLiu/Omni-Woodpecker>
- **论文：** <https://arxiv.org/abs/2610.27103>（[论文归档](../papers/omni_woodpecker_arxiv_2610_27103.md)）
- **许可：** MIT；模型权重和数据遵循各自许可。
- **核查日期：** 2026-10-28；官方推理代码、样例及配置已公开。

## 复现入口与资源

README 使用 Python 3.10、`pip install -r requirements.txt` 和 `pip install -e .`。命令入口 `src/owp_infer.py` 读取 `sample_data/owp_input.jsonl`，经 `--model videollama2` 或 `--model qwen` 选择骨干；Python API 入口是 `owp.correct`。配置在 `configs/`，干预逻辑在 `src/intervention.py` 和 `src/intervention_videollama2.py`。官方说明一次运行包括四次诊断前向和一次干预前向。

模型权重首次从 Hugging Face 下载或通过 `--model-path` 指向本地。README 要求 BF16 至少 24 GB GPU 显存；AVHBench 媒体需要本地文件或指定数据集，CMM 媒体可自动下载。仓库包含三条 CMM 样例，完整基准需另备数据。

**对 wiki 的映射：** [Omni-Woodpecker 实体页](../../wiki/entities/paper-omni-woodpecker.md)。
