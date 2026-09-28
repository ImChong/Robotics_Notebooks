# FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence（arXiv:2609.17210）

> 来源归档（ingest）

- **标题：** FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence
- **类型：** paper / platform
- **来源：** arXiv abs / PDF；项目页、产品页与 GitHub 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2609.17210>（2026-09-15）
  - PDF：<https://arxiv.org/pdf/2609.17210>
  - 项目文档：<https://fluxvla.limxdynamics.com/>（[中文](https://fluxvla.limxdynamics.com/zh/)）
  - 产品页：<https://www.limxdynamics.com/zh/products/fluxvla>
  - 代码：<https://github.com/FluxVLA/FluxVLA>
  - HF 权重：<https://huggingface.co/limxdynamics/FluxVLAEngine>
  - Zenodo：<https://doi.org/10.5281/zenodo.20049506>
- **作者：** Yinhao Li, Weixin Mao, Zihan Lan, Jikun Rong, Qirui Hu, Yiming Zhang, Weipeng Deng, Bowen Shen, Minzhao Zhu, Yiming Mao, Yan Yang, Chenguang Cui, Hongyuan Chen, Xu Huang, Zheyi Zhao, Pinxi Shen, Bozhen He, Zhen Fu, Yifan Wang, Zexin Zhang, Ang Gao, Haoyu Chen, Chengqi Shi, Hua Chen（CITATION.cff 列举；arXiv 页显示 24 作者）
- **机构：** 逐际动力（LimX Dynamics）
- **入库日期：** 2026-09-28
- **复核日期：** 2026-09-28（对照 arXiv 摘要、README、产品页与 `FluxVLA/FluxVLA` 仓库结构）
- **一句话说明：** **配置驱动的开源 VLA 工程平台**（非新策略模型）：统一数据集、VLM/世界模型、动作头、优势加权学习、分布式训练、仿真评测、加速推理与机器人 operator 接口；串联离线学习、仿真验证、在线纠正与真机执行，并集成双臂组合仿真、自动数据生成与模型解耦的人在环采集/接管/奖励标注；真机侧强调 **RTC**、远程 GPU 服务与轨迹后处理。

## 开源状态（项目页 / README 核查 2026-09-28）

- **已开源：** 主仓 [`FluxVLA/FluxVLA`](https://github.com/FluxVLA/FluxVLA)（Python，`fluxvla/` 包；`scripts/train.py` / `eval.sh` / `inference_real_robot.py`）；HF 组织 [`limxdynamics/FluxVLAEngine`](https://huggingface.co/limxdynamics/FluxVLAEngine) 发布多 backbone LIBERO / RoboCasa / RoboDojo 微调权重；文档站与 Docker Orin 镜像见 README。
- **生态子项目（README News）：** [FluxDAgger](https://github.com/FluxVLA/FluxDAgger)（模型解耦 DAgger）、[FluxBisim](https://github.com/FluxVLA/FluxBisim)（双臂操作仿真基准）；与主仓分工，非单一 monolith。
- **部分能力边界：** 各策略族（π0/π0.5、GR00T N1.5/N1.7、Cosmos3、FastWAM、DiT4DiT、DreamZero、SmolVLA 等）成熟度以 README Feature 与 `configs/` 为准；真机 ROS Noetic / Oli 全身路径需 `real-only` 或 `full` 安装档。
- **互指：** [`sources/sites/limx-fluxvla-product.md`](../sites/limx-fluxvla-product.md) · [`sources/repos/fluxvla.md`](../repos/fluxvla.md)

## 核心论文摘录（MVP）

### 1) 问题：算法繁荣 vs 工程碎片化

- **链接：** <https://arxiv.org/abs/2609.17210> 摘要 §1
- **摘录要点：** VLA、世界–动作模型（WAM）与离线 RL 快速扩展策略设计空间，但 **数据格式、训练栈、评测协议、推理运行时与本体接口** 各自为政，阻碍把算法变成可依赖的机器人系统。
- **对 wiki 的映射：**
  - [FluxVLA Engine（实体）](../../wiki/entities/fluxvla-engine.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 2) 定位：平台而非新模型

- **链接：** arXiv 摘要；README Framework 图
- **摘录要点：** FluxVLA **不提出新的策略架构**，而是以 **配置驱动** 的标准化接口连接：数据集、视觉–语言与世界模型、动作头、奖励/优势加权学习、分布式训练、仿真评测、优化推理与 robot operators；目标是把异构组件收进 **可复现、可审计** 的数据到部署工作流。
- **对 wiki 的映射：**
  - [FluxVLA Engine](../../wiki/entities/fluxvla-engine.md)
  - [LeRobot](../../wiki/entities/lerobot.md) — 数据格式与生态互操作对照

### 3) 闭环能力：仿真、自动数据、人在环

- **链接：** arXiv 摘要；README Latest News（FluxBisim / FluxDAgger）
- **摘录要点：** 集成 **组合式双臂仿真**、可扩展 **自动数据生成**、**模型解耦** 的人在环 rollout / takeover / 纠正采集与 **奖励标注**；离线学习、仿真验证、在线纠正与真机执行通过 **共享契约** 串联。
- **对 wiki 的映射：**
  - [FluxVLA Engine](../../wiki/entities/fluxvla-engine.md)
  - [VLA 部署 Query](../../wiki/queries/vla-deployment-guide.md)

### 4) 真机执行：RTC 与推理加速

- **链接：** README（RTC、Triton、ZMQ 远程推理、Jetson Orin 7.4 Hz GR00T-N1.5）
- **摘录要点：** 为响应式物理执行，组合 **Real-Time Chunking (RTC)**、加速推理后端、轻量 **远程 GPU 服务** 与可配置 **轨迹后处理**；README 报 GR00T-RTC 在 RTX 5090 上 **45 Hz**，π0.5 Triton RTC 与 ZMQ 端侧卸载。
- **对 wiki 的映射：**
  - [FluxVLA Engine](../../wiki/entities/fluxvla-engine.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)

### 5) 基准结果（README 表格，非论文主表替代）

- **链接：** README Performance — LIBERO / RoboCasa GR1 / RoboDojo
- **摘录要点：**
  - **LIBERO Average：** 多 backbone 微调；例：FluxVLA(π0.5) **97.95%**，FluxVLA(DiT4DiT) **98.65%**（链至 HF checkpoint）。
  - **RoboCasa GR1（50 trials/task）：** FluxVLA(DiT4DiT) 四组平均 **57.25%**；π0.5 **51.42%**。
  - **RoboDojo（2100 trials/model）：** FluxVLA(π0.5) 平均 progress/success **13.61% / 8.83%** vs SmolVLA **5.29% / 2.98%**——长程泛化仍难，平台提供统一评测入口。
- **对 wiki 的映射：**
  - [FluxVLA Engine](../../wiki/entities/fluxvla-engine.md)
  - [VLA 部署 12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/fluxvla-engine.md`](../../wiki/entities/fluxvla-engine.md) — 主实体页（arXiv:2609.17210 唯一详情节点，复用工程实体）
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — VLA 工程栈选型
- [`wiki/entities/limx-cosa.md`](../../wiki/entities/limx-cosa.md) — 产品层 OS vs 开源技能层分工
- [`wiki/queries/vla-deployment-guide.md`](../../wiki/queries/vla-deployment-guide.md) — RTC / 远程推理 / 端侧部署
