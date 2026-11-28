# ShowNow 发布代码｜ShowNow Release Repository

- **类型：** repo
- **代码：** https://github.com/showlab/ShowNow
- **论文：** https://arxiv.org/abs/2611.34693
- **权重：** https://huggingface.co/songyiren/ShowNow-5B-8F
- **数据：** https://huggingface.co/datasets/songyiren/ShowNow-MultiFuture
- **版本：** 固定提交 [132e178](https://github.com/showlab/ShowNow/tree/132e1780069c90a3be12c566336506ab757e0d9e)，2026-11-28 核对；源码许可 Apache-2.0。
- **核对范围：** 官方 README，`shownow/demo/generate.py`、`shownow/latency/tree_opt.py` 的实际生成、选支、融合提交路径；权重和数据入口按官方 README 记录。
- **通读：** 已完成
- **wiki：** [ShowNow 唯一详情](../../wiki/entities/paper-shownow.md)

## 静态源码核对

- [离线示例](https://github.com/showlab/ShowNow/blob/132e1780069c90a3be12c566336506ab757e0d9e/shownow/demo/generate.py)：读配置，装载场景、历史和模型，初始化动作；`RTStepper` 每轮生成八个候选，随后选择脚本给定的一个，用 TAEHV 解码并保存视频。
- [优化引擎](https://github.com/showlab/ShowNow/blob/132e1780069c90a3be12c566336506ab757e0d9e/shownow/latency/tree_opt.py)：融合提交时 `choose()` 推进索引并设待写标记，`replay_gen()` 据此选择下一轮的融合 CUDA 图；非融合时在选择后跑独立提交图。
- 发布物是 5B 八分支学生与对应代码。README 列出 10,241 个 held、11,471 个 switch、252 个 eval fork；原始 1280×720、32 fps，训练预处理到 864×480、16 fps。原始历史三秒，论文方法使用裁剪后的历史。最终 EMA 可用不表示中间检查点也公开。
- 本次没有下载大权重/数据或执行示例。仓库的离线 `generation_ms` 不含显示调度，不能据此复称论文输入到显示延迟。
