# ctp-robotics/CTP

> 来源归档

- **标题：** Contact Trajectory Prompting 官方代码
- **类型：** repo
- **代码：** <https://github.com/ctp-robotics/CTP>
- **论文：** [arXiv:2610.18848](https://arxiv.org/abs/2610.18848) — 归档见 [`sources/papers/ctp_arxiv_2610_18848.md`](../papers/ctp_arxiv_2610_18848.md)
- **项目页：** <https://ctp-robotics.github.io/project_page/> — 归档见 [`sources/sites/ctp-project.md`](../sites/ctp-project.md)
- **入库日期：** 2026-10-17（核查日同日）
- **许可：** 页面未显示（未核实）
- **一句话说明：** 两阶段训练（接触编码器预训练 → 带动作条件力预测损失的策略训练）+ 开环评测。

## 目录与入口（README 与根目录列表，未本地运行）

- 根目录：`config/`、`datasets/`、`docs/`、`models/`、`scripts/`、`tests/`、`tools/`、`trainers/`、`utils/`、`train.py`、`requirements.txt`、`README.md`。
- 数据：Zarr replay buffer（相机图像、机器人状态与动作、指尖力 / 力矩、每指触觉位移场）分 train / val；原始记录用 `tools/convert_dataset.py` 转换。
- 流程：`python train.py --config config/pretrain.yaml` → 把策略配置里的参考编码器检查点指向预训练结果 → `python train.py --config config/policy.yaml` → `python tools/eval_open_loop.py`。
- 可选：多 GPU、图像特征缓存、合成数据 CPU smoke test。
- 未见：权重、releases、真机运行脚本（核查未发现，**未核实**其不存在）。

## 与本仓库知识的关系

[CTP 实体页](../../wiki/entities/paper-ctp.md)
