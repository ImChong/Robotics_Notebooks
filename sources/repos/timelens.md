# TimeLens 官方仓库

- **链接：** <https://github.com/TencentARC/TimeLens>
- **核查日期：** 2026-10-30
- **论文：** <https://arxiv.org/abs/2512.14698>
- **项目页与榜单：** <https://timelens-arc-lab.github.io/#leaderboard>
- **数据：** <https://huggingface.co/datasets/TencentARC/TimeLens-Bench>
- **开放状态：** README 提供模型、训练及评测代码和数据入口。

## 已核对的评测入口

`scripts/eval_timelens_bench.sh` 启动 `evaluation/eval_dataloader.py`；后者以 `timelens/dataset/timelens_data.py` 加载标注、模型生成后抽取时间区间并写 JSONL。脚本合并分片后调用 `evaluation/compute_metrics.py` 输出 R1 与 mIoU。源码核查范围仅为上述文件，训练链路尚未逐模块核查。

## 对 wiki 的映射

- [TimeLens](../../wiki/entities/paper-timelens.md)
