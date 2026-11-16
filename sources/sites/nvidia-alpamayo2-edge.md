# NVIDIA Alpamayo 2 Edge — 官方发布与模型卡

> 来源归档

- **标题：** Alpamayo 2 Edge — NVIDIA in-vehicle reasoning VLA
- **类型：** site / 官方博客 / 模型卡
- **机构：** NVIDIA
- **产品页：** <https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/>
- **技术博客：** <https://developer.nvidia.com/blog/running-av-reasoning-models-within-onboard-compute-and-latency-limits/>
- **模型卡：** <https://huggingface.co/nvidia/Alpamayo2-Edge>
- **代码：** <https://github.com/NVlabs/alpamayo2-edge>
- **部署配方：** <https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo2edge_edgellm>
- **入库日期：** 2026-11-12
- **版本：** NVIDIA 2026-11-10 技术博客；Hugging Face Alpamayo2-Edge 模型卡（截至 2026-11-12）。
- **核对范围：** 通读 NVIDIA Developer Blog「Running AV Reasoning Models within Onboard Compute and Latency Limits」正文（架构、adaptive reasoning、闭环 RL、Thor 实测与局限）及 Alpamayo2-Edge Hugging Face 模型卡全文（Model Details、Inputs/Outputs、Training、Evaluation、Usage、Limitations/License），核对官方 Alpamayo 产品页入口；未下载权重或数据、未执行 GPU 真机验证。
- **通读：** 已完成
- **一句话说明：** 基于 Cosmos 3 Edge/Nemotron 的紧凑型自动驾驶推理 VLA，以自适应因果链与扩散动作专家输出未来 64 个车体位姿。
- **沉淀到 wiki：** [Alpamayo 2 Edge](../../wiki/entities/alpamayo2-edge.md)

## 官方技术事实

| 维度 | 一手资料结论 |
|---|---|
| 时间 | 2026-10-21 发布消息，2026-11-10 权重与官方技术详解公开 |
| 参数口径 | 官方称「2B」；模型卡分列 2.45B backbone（其中 1.96B language、0.41B vision、0.08B projector）和 0.2B action expert，不宜混写成精确 2B 总参数 |
| 输入 | 6 路摄像头共 22 帧、20 个自车历史位姿、自然语言导航 |
| 处理 | Cosmos 3 Edge VLM 按需输出 Chain-of-Causation（CoC）或简略 NOMINAL 分支，扩散式动作专家预测加速度/曲率 |
| 输出 | 64 个未来位姿；间隔 0.1 s、总预测区间 0.1–6.4 s；包括平移 XYZ 与姿态矩阵 |
| 训练 | 模型卡披露训练使用约 115,000 小时驾驶数据和 370 万条 CoC 轨迹；并利用 AlpaGym 闭环强化学习 |
| 评测 | 官方博客称相对最优 RL 前检查点闭环改进 48.9%；模型卡列 PhysicalAI-AV open-loop 1,392 样本 minADE_6@5s=0.52m 和 AlpaSim 875 场景测试结果 |
| 部署 | 博客给出 DRIVE AGX Thor 优化推理 160–190 ms；属于其指定硬件/软件栈下结果，非通用实时保证 |
| 许可 | 推理源码 Apache-2.0；模型权重 OpenMDW-1.1 |

## 开源边界（步骤 2.5）

- **已开源：** [推理仓库](https://github.com/NVlabs/alpamayo2-edge)、[模型权重](https://huggingface.co/nvidia/Alpamayo2-Edge)、[SFT 配方](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo2edge_sft)、[Thor 部署配方](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo2edge_edgellm)。
- **数据限制：** PhysicalAI-Autonomous-Vehicles 数据集须满足 Hugging Face 访问/许可条款；官方训练数据总量不等于全部公开可下载。
- **安全边界：** 模型卡强调这是研究/开发参考模型；需要独立的系统验证和安全约束，不能仅凭开放环指标宣称可用于安全关键车辆闭环控制。
- **跨版本说明：** 产品家族包括 Alpamayo 2 Super；Alpamayo 2 Edge 是单独的车端模型，不能直接把 34B Super 的实验结果移植到 Edge。

## 对 wiki 的映射

- 主实体：[Alpamayo 2 Edge](../../wiki/entities/alpamayo2-edge.md)
- 官方代码归档：[nvidia-alpamayo2-edge](../repos/nvidia-alpamayo2-edge.md)
- 底座：[Cosmos 3](../../wiki/entities/cosmos-3.md)
- NVIDIA 平台：[NVIDIA Cosmos](../../wiki/entities/nvidia-cosmos.md)
