# VasPug/robot-world-model-study-

> 来源归档（ingest）

- **类型：** repo
- **项目 / 代码仓：** <https://github.com/VasPug/robot-world-model-study->
- **论文 PDF：** <https://github.com/VasPug/robot-world-model-study-/blob/main/paper.pdf>
- **作者：** Aravind Kanappan、Vasanth Pugalenthi
- **入库日期：** 2026-10-12
- **一句话说明：** 模拟导航与 Push-T 控制中的世界模型咨询对照实验、协议、脚本与参考结果。

## 运行入口

- 依赖：Python `>=3.11,<3.12`；`uv sync --locked`
- 结果核验：`uv run --locked python scripts/verify_results.py`
- CPU 参考统计检查：`uv run --locked python scripts/analyze_gpu.py --check-reference`
- CPU 导航：`bash run_navigation.sh`
- Push-T pilot：`bash run.sh --config configs/pusht/baseline.json` 或 `bash run.sh --config configs/pusht/world_model_b.json`
- GPU 扩展实验：`bash run_gpu.sh`（需要 CUDA）

## 代码 / 结果边界

仓库包含论文、协议文档、脚本与结果摘要。核验脚本用于对照固定参考结果，不代表重跑全套实验；GPU 扩展需 CUDA。实证结果来自模拟器，不等于真实机器人验证。

## 关联来源

- [论文归档](../papers/world-model-consultation-study.md)
- [实体页](../../wiki/entities/paper-world-model-consultation-study.md)
- [Model-Based RL](../../wiki/methods/model-based-rl.md)
