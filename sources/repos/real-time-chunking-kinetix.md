# real-time-chunking-kinetix

> 来源归档（repo）

- **标题：** real-time-chunking-kinetix
- **类型：** repo
- **链接：** <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **关联论文：** [real_time_chunking_arxiv_2506_07339.md](../papers/real_time_chunking_arxiv_2506_07339.md)
- **入库日期：** 2026-09-28
- **一句话说明：** RTC 论文的 Kinetix 仿真实验：专家数据、flow 模仿、延迟扫描，以及 training-time RTC 微调。

## 开源状态

- **部分开源**：仿真训练与评测脚本在仓内；专家与 BC 检查点、百万转移数据放在 `gs://rtc-assets/`（README 写 `expert/` 合计约 60GiB）。
- 真机 π 策略的 RTC 推理不在本仓。

## README 入口

1. 专家：`src/train_expert.py`（或直接用 `gs://rtc-assets/expert/`）。
2. 数据：`src/generate_data.py`。
3. 模仿：`uv run src/train_flow.py`。
4. 评测：`uv run src/eval_flow.py`，默认对推理延迟和 execution horizon 做扫描。
5. Training-time RTC：模型配置里 `simulated_delay = 5`，再对预训练 BC 检查点微调 8 个 epoch。

## 对 wiki 的映射

- [paper-real-time-chunking](../../wiki/entities/paper-real-time-chunking.md)
