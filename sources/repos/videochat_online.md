# VideoChat-Online 官方源码（MCG-NJU）

- **仓库：** https://github.com/MCG-NJU/VideoChat-Online
- **论文：** https://arxiv.org/abs/2501.00584
- **项目页：** [videochat-online](../sites/videochat-online-project.md)
- **论文源：** [OVBench](../papers/ovbench_videochat_online_arxiv_2501_00584.md)
- **Wiki：** [OVBench / VideoChat-Online](../../wiki/entities/paper-ovbench-videochat-online.md)
- **入库日期：** 2026-11-16

## 真实代码入口

| 入口 | 作用 |
|---|---|
| `shell/sft/videochat_online_4b_stage1_ft.sh` | 离线 SFT 训练 |
| `shell/sft/videochat_online_4b_stage2_ft.sh` | 在线与离线联合 SFT |
| `shell/eval/online_bench_stream.sh` | `torchrun` 启动 `eval/evaluate_online_stream_single.py` |
| `shell/eval/online_bench_sliding_window.sh` | 滑动窗口评测 |
| `eval/evaluate_online_stream_single.py` | decord 解码、时间点 QA、模型调用和输出日志 |
| `internvl/model/videochat_online` | 模型实现 |
| `gradio_demo.sh` | 调用 `internvl/serve/app.py` |

## 复现提示

- README 的 `shell/online_4b/` 训练命令已不是当前现存目录；应使用 `shell/sft/`。
- `online_bench_stream.sh` 包含 `/workspace/data/...` 硬编码路径、默认 `NUM_GPUS=8` 和训练权重目录 `work_dirs/VideoChatOnline_Stage2`，运行前需要本地修改。
- PMB 可读取 ViT 特征实现；KV-cache 版本不宜宣称已发布。未实际运行训练或评测。
