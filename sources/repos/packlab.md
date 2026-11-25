# PackLab 官方仓库

> 来源归档（repo）

- **标题：** PackLab: A Comprehensive Framework for Developing, Training, and Evaluating MLLMs in Robotic Bin Packing
- **类型：** repo
- **链接：** https://github.com/Correr-Zhou/PackLab
- **arXiv：** <https://arxiv.org/abs/2609.23784>
- **入库日期：** 2026-10-18
- **最后核对：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 核对 `Correr-Zhou/PackLab` 仓库 README、`train/launch_verl.py`、`eval/vllm_runner.py`（`rollout_case_with_client`、`PackingEnv`、`PolicyLLMJson`、`PackingScore`、`EpisodeControl`）
- **一句话说明：** 把仿真平台、专用多模态模型和分难度评测连在一起，让模型随箱内状态更新选择物体与落点。
- **沉淀到 wiki：** [`wiki/entities/paper-packlab.md`](../../wiki/entities/paper-packlab.md)

## 开源状态

- **已开源**（2026-11-21 复核）：`Correr-Zhou/PackLab`（`main` 分支）提供 `train/launch_verl.py` 训练启动与 `eval/vllm_runner.py` 闭环仿真评测脚本。
