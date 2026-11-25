---
type: entity
tags:
  - paper
  - vla
  - platform
  - deployment
  - open-source
  - limx
  - training
  - inference
  - pi0
  - groot
  - rtc
  - lerobot
  - manipulation
  - humanoid
status: complete
updated: 2026-11-22
arxiv: "2609.17210"
code: https://github.com/FluxVLA/FluxVLA
original:
  - ../../sources/papers/fluxvla_arxiv_2609_17210.md
  - ../../sources/repos/fluxvla.md
  - ../../sources/models/fluxvla-engine-huggingface.md
  - ../../sources/sites/limx-fluxvla-product.md
related:
  - ./limx-cosa.md
  - ../methods/vla.md
  - ../methods/action-chunking.md
  - ../entities/lerobot.md
  - paper-openvla.md
  - ../entities/lingbot-vla.md
  - ../queries/vla-deployment-guide.md
  - ../overview/vla-deploy-12-papers-technology-map.md
  - ../overview/vla-open-source-repro-landscape-2025.md
  - ../tasks/loco-manipulation.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/fluxvla_arxiv_2609_17210.md
  - ../../sources/repos/fluxvla.md
  - ../../sources/models/fluxvla-engine-huggingface.md
  - ../../sources/sites/limx-fluxvla-product.md
  - ../../sources/blogs/limx_cosa_05_release_2026-07-15.md
summary: "FluxVLA Engine（arXiv:2609.17210，逐际动力）：配置驱动的开源 VLA 工程平台——统一数据/训练/仿真评测/RTC 推理/真机 operator，即插 π0.5/GR00T/Cosmos3/FastWAM/DiT4DiT 等；非新策略模型；HF 权重 + FluxDAgger/FluxBisim 生态。"
---

# FluxVLA Engine（arXiv:2609.17210）

**FluxVLA Engine**（*A One-Stop VLA Engineering Platform for Embodied Intelligence*，[arXiv:2609.17210](https://arxiv.org/abs/2609.17210)，[PDF](https://arxiv.org/pdf/2609.17210)，[代码](https://github.com/FluxVLA/FluxVLA)，[文档](https://fluxvla.limxdynamics.com/zh/)，[产品页](https://www.limxdynamics.com/zh/products/fluxvla)，**逐际动力 LimX Dynamics**）是 **配置驱动** 的开源 **VLA 工程平台**：**不提出新的策略架构**，而是用标准化接口把数据集、VLM/世界模型、动作头、优势加权学习、分布式训练、仿真评测、加速推理与 robot operator 收进 **可复现、可审计** 的数据→部署闭环。它与 [LimX COSA](./limx-cosa.md) 的分工是：**COSA = 闭源大脑 OS 与多技能调度；FluxVLA = 开源 S1 技能层的训练、评测、反馈与端侧推理底座**。

## 一句话定义

**把「能发论文的 VLA/WAM/离线 RL 组件」接到同一套配置与契约上，从 LeRobot 兼容数据一路跑到 LIBERO/RoboCasa 评测、RTC 推理与 Franka/Oli 真机——工程瓶颈优先于再训一个新 backbone。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 平台核心策略对象（多 backbone 可切换） |
| WAM | World-Action Model | 世界–动作模型；与 VLA 同属平台接入的策略族 |
| RTC | Real-Time Chunking | 跨 chunk 低延迟连续执行（π0.5/GR00T 等路径） |
| BC | Behavior Cloning | 模仿学习微调常见起点 |
| ARM | Advantage-weighted Reward Model | 平台内 RA-BC/AW-BC 奖励重加权管线 |
| HF | Hugging Face | 官方 checkpoint 组织 `limxdynamics/FluxVLAEngine` |

## 核心信息

| 字段 | 内容 |
|------|------|
| **机构** | 逐际动力（LimX Dynamics） |
| **arXiv** | [2609.17210](https://arxiv.org/abs/2609.17210)（2026-09-15） |
| **开源** | **已开源**（2026-10-28 复核）— [`FluxVLA/FluxVLA`](https://github.com/FluxVLA/FluxVLA)；HF [`limxdynamics/FluxVLAEngine`](https://huggingface.co/limxdynamics/FluxVLAEngine)；子项目 [FluxDAgger](https://github.com/FluxVLA/FluxDAgger)、[FluxBisim](https://github.com/FluxVLA/FluxBisim) |
| **安装** | `bash scripts/install_env.sh`（`sim-only` / `real-only` / `full`）；Python 3.10 |
| **真机** | Franka 单/双臂；Oli 人形全身 loco-manipulation 最小路径（ROS + WebSocket）；Jetson Orin Docker |

## 为什么重要

- **第三类 VLA 价值：** 相对「又一个刷榜模型」或 [Harness VLA](./paper-harness-vla.md) 式 **冻结策略 + Agent 编排**，FluxVLA 押 **训练栈、评测口径、推理运行时与本体接口** 的标准化——适合要 **切换 backbone、审计实验、推进部署** 的团队。
- **与 [LeRobot](./lerobot.md) 互补：** LeRobot 偏 **通用数据/策略格式**；FluxVLA 偏 **多 VLA/WAM 后端 + LimX 人形/双臂真机栈** 的统一 `configs/` 与 `scripts/`（二者可交叉，勿混为同一项目）。
- **部署证据链完整：** README 同时给出 **LIBERO 数字**、**RoboCasa GR1**、**RoboDojo 长程** 与 **真机 Demo GIF**；RTC + Triton + ZMQ 远程 GPU 对齐 [VLA 部署](../queries/vla-deployment-guide.md) 与 [Action Chunking](../methods/action-chunking.md) 主线。
- **国内开源地图锚点：** arXiv:2609.17210 在本库 **唯一详情节点即本页**（工程实体复用，不另建 `paper-fluxvla-*` 重复页）。

## 流程总览

```mermaid
flowchart TB
  subgraph data [数据层]
    LER[LeRobot / HDF 转换]
    AUTO[自动数据生成]
    HITL[人在环 rollout / 纠正 / 奖励标注]
  end
  subgraph train [训练层]
    CFG[configs/*.py 统一配置]
    ENG[fluxvla engines + 分布式 torchrun]
    WGT[ARM / AW-BC 样本重加权]
  end
  subgraph val [验证层]
    LIB[LIBERO / RoboCasa / RoboDojo]
    BIS[FluxBisim 双臂仿真]
  end
  subgraph deploy [部署层]
    RTC[RTC + Triton / CUDA Graph]
    ZMQ[ZMQ 远程 GPU 服务]
    OP[robot operator / 轨迹后处理]
    EDGE[Jetson Orin 端侧]
  end
  LER --> CFG
  AUTO --> LER
  HITL --> LER
  CFG --> ENG --> WGT
  ENG --> LIB
  ENG --> BIS
  LIB --> RTC
  RTC --> ZMQ
  RTC --> OP
  ZMQ --> EDGE
  OP --> EDGE
```

## 源码运行时序图

对齐 [`FluxVLA/FluxVLA`](https://github.com/FluxVLA/FluxVLA) README：`scripts/train.py` / `eval.sh` / `inference_real_robot.py` 与 `fluxvla/` 包分层。

```mermaid
sequenceDiagram
  autonumber
  actor U as 用户 / 集群脚本
  participant TR as scripts/train.py<br/>torchrun
  participant ENG as fluxvla.engines
  participant MD as fluxvla.models
  participant DS as fluxvla.datasets
  participant EV as scripts/eval.sh<br/>fluxvla.evaluators
  participant SIM as LIBERO / RoboCasa
  participant INF as inference_real_robot.py<br/>或 ZMQ 推理服务
  participant ROB as Franka / Oli operator
  U->>TR: --config configs/...py --work-dir
  TR->>DS: 加载 LeRobot 兼容 batch
  TR->>ENG: 分布式 step / checkpoint
  ENG->>MD: 前向 + 动作头 / WAM 头
  U->>EV: eval.sh CONFIG CKPT
  EV->>MD: 加载权重
  EV->>SIM: rollout + 指标汇总
  U->>INF: --config + ckpt-path
  INF->>MD: RTC 解码 action chunk
  INF->>ROB: 关节/笛卡尔/全身命令
  ROB-->>INF: 传感器反馈（闭环）
```

- **最短复现路径：** `install_env.sh sim-only` → 选 README Performance 表对应 `configs/` → `torchrun scripts/train.py` → `eval.sh` → HF checkpoint 可跳过自训；在线 RL 后训练可用 `rlinf-only` 模式。
- **真机路径：** `real-only` 或 `full` → `docs/franka.md` / `docs/oli_whole_body.md` → `inference_real_robot.py`；远程算力见 `docs/remote_inference_serving.md`。

图依据：`FluxVLA/FluxVLA@b401e52`，读过 `README.md`、`scripts/train.py` 与 `scripts/inference_real_robot.py`（`Config.fromfile`、`build_runner_from_cfg`、`inference_runner.run_setup/run`），未实际运行训练或真机推理。

## 支持的策略与评测（README 摘要）

| 策略族 | 平台角色 |
|--------|----------|
| **π0 / π0.5** | Flow-matching VLA；LIBERO π0.5 平均 **97.95%**；Triton RTC 后端 |
| **GR00T N1.5 / N1.7** | NVIDIA 人形 VLA；RTC **45 Hz**（5090）；Orin **7.4 Hz** 报道 |
| **Cosmos3 Nano/Super/Edge** | 纯视觉/post-train 路径；Edge LIBERO 平均 **94.4%** |
| **FastWAM / FastWAM-IDM/Joint** | 世界–动作模型族；LIBERO 最高 **98.35%** 量级 |
| **DiT4DiT** | LIBERO **98.65%**；RoboCasa GR1 **57.25%** |
| **DreamZero / SmolVLA / OpenVLA / LlavaVLA** | 文档与 `configs/` 接入；成熟度以 Feature 矩阵为准 |

**RoboDojo 提醒：** π0.5 平均 progress/success **13.61% / 8.83%** 仍显著低于 LIBERO  tabletop 分数——长程泛化是 **平台能测、模型仍难** 的显例。

**结果口径：** GitHub README 报 π0.5 LIBERO 四套件平均 97.95%，[HF 模型卡](../../sources/models/fluxvla-engine-huggingface.md)报 97.85%；两个来源的分项也不同，不能直接视为同一次实验或取平均。选定 checkpoint 和评测配置后再复现。

### 论文实证补充（arXiv:2609.17210v2）

- **RoboTwin 与在线 RL 后训练（RLinf）：** 在 RoboTwin 50 任务（每任务 50 clean + 500 randomized）多任务训练中，FastWAM Clean/Random 达 `90.86% / 89.96%`（平均 `90.41%`）；可选 **RLinf** PPO/GAE 前端在不改原生预处理的前提下，将 SmolVLA 在 LIBERO-10 从 `66.2%` 提至 `76.2%`（+10.0 pt）、在 10 项 RoboTwin Clean 任务从 `45.9%` 提至 `61.3%`（+15.4 pt）。
- **推理加速数值对齐与轨迹后处理：** GR00T 加速路径（算子融合 + CUDA Graph）相对基线随机权重余弦相似度 `> 0.99999`（最大绝对误差约 `0.02`），RTX 5090 达 `42.6 Hz`（Training-time RTC `47.6 Hz`）、AGX Orin 64GB 达 `7.4 Hz`；12-DoF 50 步合成轨迹上，Ruckig 滤波主机端均值延迟 `1.17 ms`、Joint MPC 为 `4.90 ms`。
- **真机多平台评测（Project-reported）：**
  - **ALOHA 6 任务（90 trials/模型）：** `π0.5` 汇总成功率 `72.22%`，`GR00T N1.7` `62.22%`，`FastWAM` `61.11%`。
  - **Oli 人形 4 任务：** `π0.5` 糖果拾取 `62.5%`、桌面清理 `45.0%`、4 阶段搬箱全流程 `33.33%`（1–3 阶段 `61.67%`）。
  - **Rebot 4 任务（80 trials）：** `π0.5` `77.50%`；**双臂 Franka 5 任务（100 trials）：** `GR00T N1.7` `88.00%`、`π0.5` `79.00%`。
  - **生态分工规划（arXiv §6.2）：** 围绕稳定契约拆分为 `FluxMimir`（离线数据）→ `FluxBisim`（仿真与数据生成）→ `FluxDAgger`（人在环干预）→ `FluxVLA Engine`（训推核心）→ `FluxThemis`（仿真/真机评测编排）→ `FluxHermes`（编译与部署打包）。

## 结论

**FluxVLA Engine 的核心工程价值不是「新刷榜模型」，而是把异构 VLA/WAM/RL 的数据、配置、训练、评测、RTC 与真机执行接口收敛到同一套可审计契约上。**

1. **做跨模型对照时先锁契约再看数字**：LIBERO/RoboCasa/RoboTwin 表格训练预算与微调条件并不对齐，跨行比较须用同一 `configs/` 与数据配方重跑。
2. **模型推理频率 ≠ 机器人闭环频率**：5090 上的 `40+ Hz` 仅代表模型前向路径，端到端闭环仍受相机采集、ZMQ 序列化、RTC 衔接与轨迹后处理影响。
3. **长程泛化暴露真实瓶颈**：同一平台上 LIBERO 普遍 `>94%` 而 RoboDojo 仅 `0.93%–8.83%` 成功率，说明难点在长程记忆与随机布局泛化，而非训练框架。
4. **分清开源底座与商业 OS 边界**：需要开源 VLA 训推与端侧运行时选 FluxVLA；需要 S2 认知调度与机上 1000 Hz 运控仍依赖 COSA 或自研上层调度。
5. **用子系统边界管理复杂度**：结合 `FluxBisim`、`FluxDAgger` 与 `FluxThemis` 分仓演化，避免在主仓耦合全部仿真器与评测脚本。

## 工程实践

| 项 | 说明 |
|----|------|
| 数据 | `tools/convert_hdf_to_lerobot.py`；与 LeRobot schema 对齐 |
| 奖励 | `tools/arm_awbc`、`docs/arm.md` — RA-BC/AW-BC |
| 在线纠正 | [FluxDAgger](https://github.com/FluxVLA/FluxDAgger) 模型解耦 DAgger |
| 仿真 | [FluxBisim](https://github.com/FluxVLA/FluxBisim)；RoboCasa 资产 `download_robocasa_assets.py` |
| 推理 | `docs/rtc.md`、`docs/inference_acceleration.md`、ZMQ 远程服务 |
| 端侧 | `fluxvla/fluxvla-orin` Docker；`docs/orin_docker_runtime.md` |

## 复现要点（论文 + 工程读法）

1. **先选层：** 要对比 **backbone** → 用同一 `configs/` 与 HF 权重；要验证 **部署** → 优先 RTC + 远程推理文档，而非只刷 LIBERO 分。
2. **别把平台当 COSA：** 长程家庭 loco-manipulation Demo 依赖 **上层 OS**；FluxVLA 提供 **技能训练与推理**，不含 S2 认知调度。
3. **RoboDojo vs LIBERO：** 高 LIBERO 分 **不** 自动等于长程 kitchen/generalization；读 README 双指标。
4. **环境档：** 仿真-only 勿装全量真机依赖；Orin/ROS 路径单独跟文档，避免 `full` 安装失败误判「平台不可复现」。
5. **生态子仓：** DAgger 与 Bisim 版本独立迭代，issue 时注明主仓 vs 子仓。

## 局限与风险

- **局限：** 策略族迭代快，README News 快于 arXiv v1/v2 正文；论文 §5.8 明确指出当前尚未完成全模型同预算对照、注入延迟下的 RTC 消融以及多机统一协议基准。

## 常见误区

- **误区：「支持列表 = LimX 官方维护全部上游权重」。** License 与 checkpoint 仍遵循 OpenPI、NVIDIA 等；平台提供 **接口与发布权重链接**。
- **误区：「装了 FluxVLA = 论文算法已 SOTA」。** 平台论文主张 **工程可复现**；各 backbone 数字是 **在统一栈上的对照**，不是单一模型 claim。

## 关联页面

- [LimX COSA](./limx-cosa.md) — 调度 FluxVLA 训练出的 VLA 作为 S1 技能
- [VLA 方法页](../methods/vla.md) — 策略族总览
- [Harness VLA](./paper-harness-vla.md) — 另一类「用好已有 VLA」：Agent + 冻结原语 vs 本页 **全栈训练部署**
- [LingBot-VLA](./lingbot-vla.md) — 4B 双臂基座；可作 FluxVLA 接入 backend 对照
- [VLA 部署 12 篇地图](../overview/vla-deploy-12-papers-technology-map.md) — 本期 #01 入口

## 推荐继续阅读

- [FluxVLA 中文文档](https://fluxvla.limxdynamics.com/zh/)
- [π0.5 微调示例（FluxVLA）](https://fluxvla.limxdynamics.com/zh/md_source/examples/pi0.html)
- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)

## 参考来源

- [FluxVLA Engine（arXiv:2609.17210 归档）](../../sources/papers/fluxvla_arxiv_2609_17210.md)
- [FluxVLA 源码归档](../../sources/repos/fluxvla.md)
- [FluxVLA Hugging Face 模型卡归档](../../sources/models/fluxvla-engine-huggingface.md)
- [LimX FluxVLA 产品页归档](../../sources/sites/limx-fluxvla-product.md)
- [COSA 0.5 发布：FluxVLA 同步开源](../../sources/blogs/limx_cosa_05_release_2026-07-15.md)
