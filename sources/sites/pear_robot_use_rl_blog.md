# PEAR 官方研究博客（MIT HAN Lab）

> 来源归档：官方项目介绍

- **标题：** Training Reasoning Models in Simulation to Natively Use Real Robots（在仿真中训练推理模型，使其原生操作真实机器人）
- **类型：** site（官方博客 / 项目页）
- **链接：** <https://mit-han-lab.github.io/pear-blog/>
- **论文：** <https://mit-han-lab.github.io/pear-blog/paper.pdf>
- **代码：** <https://github.com/mit-han-lab/pear>
- **模型权重：** <https://huggingface.co/collections/mit-han-lab/pear>
- **入库日期：** 2026-11-15
- **核对范围：** 官方博客 TL;DR、How it works、Robot-Use RL、鲁棒性/指令跟随、跨平台与真实机器人、通用能力、推理速度与局限、附录和资源链接。
- **通读：** 已完成
- **一句话说明：** PEAR 使用 Robot-Use RL 在仿真中让 Qwen3.8-27B 学会机器人操作；博客附演示视频与作者报告的多项基准。

## 开源核对

- 已公开：GitHub 代码、训练与评估配方、模型服务/真机 station 实现；Hugging Face 三种模型资产（标准、LIBERO-only、NVFP4）。
- 不要把官网演示理解成可直接复现全部真实机器人实验：YAM/SO-101 需实体设备、相机、CAN/串口和标定。
- 本 PEAR = **Post-training Engine for Agentic Robotics**；不是已有的 [PEAR 人体网格恢复论文](../../wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md)。

## 对 wiki 的映射

- [PEAR / RURL 论文实体](../../wiki/entities/paper-pear-robot-use-rl.md)
- [论文全文归档](../papers/pear_robot_use_rl_mit_2026.md)
- [官方仓库归档](../repos/mit_han_lab_pear.md)
