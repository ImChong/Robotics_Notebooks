# Fast-WAM 官方仓库

> 来源归档（repo）

- **标题：** Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- **类型：** repo
- **链接：** https://github.com/yuantianyuan01/FastWAM
- **arXiv：** <https://arxiv.org/abs/2603.16666>
- **入库日期：** 2026-09-23
- **核对范围：** 已克隆 `yuantianyuan01/FastWAM@7faa711`（核对 `README.md`、`scripts/train.py`、`scripts/train_zero1.sh`、`scripts/precompute_text_embeds.py`、`experiments/libero/run_libero_manager.py`、`src/fastwam/runtime.py`、`src/fastwam/models/wan22/fastwam.py`；未在本地 GPU 集群执行训练或仿真评测）。
- **通读：** 已完成
- **一句话说明：** Fast-WAM 官方原生训练/推理代码仓，基于 Wan2.2-TI2V-5B + Action DiT，支持 LIBERO/RoboTwin 2.0 与 LeRobot 2.1/3.0，并补充 Optional IDM 与编译推理加速（RTX 4090 190→110 ms、H20 470→210 ms）。
- **沉淀到 wiki：** [`wiki/entities/paper-fast-wam.md`](../../wiki/entities/paper-fast-wam.md)

## 一手仓库要点摘录（`yuantianyuan01/FastWAM@7faa711`）

- **核心模块与入口**：`src/fastwam/models/wan22/fastwam.py` 实现 `FastWAM` 类（含 `training_loss`、`infer_action`、`infer_joint`、`infer`）；训练入口 `scripts/train.py` / `scripts/train_zero1.sh`（预计算文本特征 `scripts/precompute_text_embeds.py`）；仿真评估入口 `experiments/libero/run_libero_manager.py`，通用部署封装 `src/fastwam/runtime.py`。
- **更新特性与官方 HF 资源**：支持 `yuanty/fastwam` 模型及 `yuanty/LIBERO-fastwam`、`yuanty/robotwin2.0-fastwam` 数据集；新增 Optional IDM 模式（LIBERO 98.55% vs First-frame 97.75%）及推理编译优化。

## 开源状态

- **已开源**（以 README 与 release 为准）。

## 官方资源补核（2026-10-05）

- **项目页：** [FastWAM 项目归档](../sites/fast-wam.md)，<https://yuantianyuan01.github.io/FastWAM/>。
- **代码：** <https://github.com/yuantianyuan01/FastWAM>。
- **权重与统计：** <https://huggingface.co/yuanty/fastwam>；LIBERO / RoboTwin 公开 checkpoint 与匹配 dataset_stats。
- **入口：** `scripts/precompute_text_embeds.py` 缓存文本；`scripts/train_zero1.sh` / `scripts/train.py` 训练；`experiments/libero/run_libero_manager.py`、`experiments/robotwin/run_robotwin_manager.py` 评测。
- **当前实现：** LeRobot 2.1 / 3.0、Optional IDM 路径属于仓库后续更新；复现初版论文须锁定对应配置，不能混用最新推理优化数字。
