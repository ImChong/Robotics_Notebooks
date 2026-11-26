# LDS 与 RoboTwin-IF | 官方代码归档

- **方法源码：** [RuFlect-Intelligence/Linear-Decodability-Supervision @ 0cd3465](https://github.com/RuFlect-Intelligence/Linear-Decodability-Supervision/tree/0cd3465abc0e0de84a21d32f30bd4b08dd86a50b)，Apache-2.0。
- **评测源码：** [RuFlect-Intelligence/RoboTwin-IF @ b7f860d](https://github.com/RuFlect-Intelligence/RoboTwin-IF/tree/b7f860d4d5f325f43e30015fed82759ece8cd779)，遵循仓内及各资产许可。
- **论文及项目：** [v1 论文](../papers/lds_arxiv_2611_27401.md) · [项目主页](../sites/lds-homepage.md)
- **核对范围：** 2026-11-26 核对 LDS README、RoboTwin-IF 训练 recipe、数据 Bundle、split-rank 训练和 readout/损失；核对 Mixed 50 CLI、rollout、Grounding 与报告入口。RoboTwin-IF 两项官方 Grounding 单元测试（接触阈值、连续停留）均通过；未运行 GPU 训练或仿真。
- **通读：** 已完成

- **训练入口：** README 的 `torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank` 选 OpenWAM、LingBot-VA 或 InternW0-Δ adapter；`--bundle` 需要含原生轨迹和定位 queries/masks 的受支持布局。公开 LeRobot expert episodes 不是可直接喂入的训练 bundle；官方 README 将独立包端到端 GPU 验证列为待完成。
- **评测入口：** `python -m robotwin_if validate` 核验固定 bundle，`run` 加载 `create_policy(config)` 协议并写逐 episode 结果，`report` 汇总动作成功、Grounding 并检查配对初态。
- **发布数据：** [RoboTwin-IF 文件列表](https://huggingface.co/datasets/RuFlect/RoboTwin-IF/tree/ec4e8072084eb1fb64b8df25d97c3ea2c82a5844)，作者列 275,270 条专家轨迹，CC BY 4.0；Mixed 50 资产另随原始许可。另核对 [InternW0-Δ LIBERO-CF 公开模型条目](https://huggingface.co/RuFlect/LIBERO-CF-InternW0-Delta-LDS/tree/5498e8b9e7d3f41166556e3d9f3283d6270bedb0)，未完整下载权重。
