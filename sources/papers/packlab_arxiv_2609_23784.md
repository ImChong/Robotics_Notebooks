# PackLab（arXiv:2609.23784）

> 来源归档（paper）

- **标题：** PackLab: A Comprehensive Framework for Developing, Training, and Evaluating MLLMs in Robotic Bin Packing
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.23784>
- **PDF：** <https://arxiv.org/pdf/2609.23784>
- **代码：** <https://github.com/Correr-Zhou/PackLab>
- **入库日期：** 2026-10-18
- **最后核对：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文通读 `arXiv:2609.23784`（含 `PackData-20K`、`PackLab-Bench` 60 cases / 905 steps、AUBO 真机试验）并核对官方仓库 `Correr-Zhou/PackLab`（`eval/vllm_runner.py`、`train/launch_verl.py`、`PackingEnv`、`PolicyLLMJson`、`PackingScore`、`EpisodeControl`）
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 把仿真平台、专用多模态模型和分难度评测连在一起，让模型随箱内状态更新选择物体与落点。

## 开源状态

- **已开源**（2026-11-21 复核）：`Correr-Zhou/PackLab` 公开训练与评测源码，核心入口为 `train/launch_verl.py` 与 `eval/vllm_runner.py`。

## 核心摘录（已核验论文原文与官方源码）

- 构建 `PackData-20K` 训练语料与 `PackLab-Bench`（60 个案例 / 905 步）分难度闭环装箱评测。
- `PackLab-VLM-9B` 随箱内状态更新决策，论文报告指标达到 `0.882 / 0.718 / 0.660`（对比启发式 SDF-Pack `0.602`、DBLF `0.554` 及基础模型 Qwen3.5-9B `0.209`），并在 AUBO 机械臂真机验证物理可执行性。

## 对 wiki 的映射

- [paper-packlab](../../wiki/entities/paper-packlab.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
