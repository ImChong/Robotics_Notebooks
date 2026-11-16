# real-time-chunking-kinetix

> 来源归档（repo）

- **标题：** real-time-chunking-kinetix
- **类型：** repo
- **链接：** <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **关联论文：** [real_time_chunking_arxiv_2506_07339.md](../papers/real_time_chunking_arxiv_2506_07339.md)
- **入库日期：** 2026-09-28
- **核对范围（2026-11-14）：** 阅读仓库 README；检查 `src/eval_flow.py` 中 `eval()/main()`、`src/model.py` 中 `FlowPolicy.realtime_action()` 和 `get_prefix_weights()`、`src/train_flow.py` 的数据加载及训练循环、`src/generate_data.py` 的检查点与数据入口。未完整通读全部源码，也未运行实验。
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

## 实际源码调用（2026-11-14 核验）

- **训练链：** `train_expert.py` 训练 Kinetix 专家 → `generate_data.py` 根据关卡与解题率筛选专家 checkpoint，收集 `obs/action/done` 轨迹 → `train_flow.py` 加载 `data/*.npz` 训练 `model.FlowPolicy`（流匹配动作块）。
- **评测链：** `eval_flow.main()` 装载每个关卡的 BC checkpoint；`eval()` 在每次执行视野内调用 `FlowPolicy.realtime_action()`，用 `jax.vjp` 对软掩码约束求梯度，拼接旧/新 chunk，再以 `jax.lax.scan(env.step)` 推进仿真；对 0–4 步延迟和合法 execution horizon 扫描，输出 `results.csv`。
- **源码分叉：** `model.py::realtime_action()` 的 `simulated_delay is None` 对应原论文推理期 PiGDM 引导；另一分支的 `simulated_delay` 是训练期 RTC 实验，不能混作同一算法。官方仓库的评测通过时间步模拟延迟，**不是**真机多线程控制器实现。
- **复现前提：** `git submodule update --init`、`uv sync`，默认 12 关且 GPU 数须整除关卡数；可从 `gs://rtc-assets/bc/` 获取检查点。未在本次环境执行安装、下载大体量资源或真机测试。
