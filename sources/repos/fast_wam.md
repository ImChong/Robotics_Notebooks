# Fast-WAM 官方仓库

> 来源归档（repo）

- **标题：** Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- **类型：** repo
- **链接：** https://github.com/yuantianyuan01/FastWAM
- **arXiv：** <https://arxiv.org/abs/2603.16666>
- **入库日期：** 2026-09-23
- **一句话说明：** 训练期保留视频共训、推理期跳过未来视频去噪，190 ms 延迟（>4× 快于 imagine-then-execute WAM）；LIBERO 97.6% / RoboTwin 91.8%。
- **沉淀到 wiki：** [`wiki/entities/paper-fast-wam.md`](../../wiki/entities/paper-fast-wam.md)

## 开源状态

- **已开源**（以 README 与 release 为准）。

## 官方资源补核（2026-10-05）

- **项目页：** [FastWAM 项目归档](../sites/fast-wam.md)，<https://yuantianyuan01.github.io/FastWAM/>。
- **代码：** <https://github.com/yuantianyuan01/FastWAM>。
- **权重与统计：** <https://huggingface.co/yuanty/fastwam>；LIBERO / RoboTwin 公开 checkpoint 与匹配 dataset_stats。
- **入口：** `scripts/precompute_text_embeds.py` 缓存文本；`scripts/train_zero1.sh` / `scripts/train.py` 训练；`experiments/libero/run_libero_manager.py`、`experiments/robotwin/run_robotwin_manager.py` 评测。
- **当前实现：** LeRobot 2.1 / 3.0、Optional IDM 路径属于仓库后续更新；复现初版论文须锁定对应配置，不能混用最新推理优化数字。
