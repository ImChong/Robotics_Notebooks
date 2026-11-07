---
type: entity
tags: [software, world-models, physical-ai, nvidia, wfm, sim2real, video-generation, open-source]
status: complete
updated: 2026-11-07
related:
  - ./paper-cosmos-transfer1.md
  - ./paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md
  - ./nvidia-cosmos.md
  - ./cosmos-3.md
  - ./cosmos-cookbook.md
  - ./newton-physics.md
  - ./nvidia-omniverse.md
  - ./nvidia-so101-sim2real-lab-workflow.md
  - ../methods/generative-world-models.md
  - ../concepts/sim2real.md
  - ../concepts/video-as-simulation.md
  - ../tasks/manipulation.md
original:
  - ../../sources/papers/cosmos_transfer1_arxiv_2503_14492.md
sources:
  - ../../sources/sites/company-roadmap-date-audit-2026-10-08.md
  - ../../sources/sites/cosmos-transfer1-project.md
  - ../../sources/repos/nvidia_cosmos_transfer1.md
  - ../../sources/papers/cosmos_transfer1_arxiv_2503_14492.md
  - ../../sources/sites/cosmos-transfer25-docs.md
  - ../../sources/repos/nvidia_cosmos_transfer25.md
  - ../../sources/sites/cosmos-cookbook.md
  - ../../sources/papers/cosmos_predict25_arxiv_2511_00062.md
summary: "Cosmos Transfer 是 NVIDIA 的 world-to-world 翻译族：用多模态 ControlNet（depth / edge / seg / blur，驾驶另加 HDMap / LiDAR）把仿真或真机视频译成可控照片级世界。Transfer1-7B 定义自适应时空加权；Transfer2.5-2B 更小更快。2026-06 起官方引导迁 Cosmos 3。"
code: https://github.com/nvidia-cosmos/cosmos-transfer2.5
---

# Cosmos Transfer（条件世界翻译）

**Cosmos Transfer** 是 [NVIDIA Cosmos](./nvidia-cosmos.md) 里专门做 **world-to-world** 的一支：输入不是「从零生成世界」，而是 **已有视频 + 空间控制图 + 文本**，输出结构对齐、外观可改的照片级世界。产品文档把它写成两条增广：**仿真→照片级**（不必把 3D 场景做到像素保真）和 **缩放世界状态多样性**（同一几何换天气、材质、时段）。

## 一句话定义

**用多 ControlNet 把仿真或真机视频译成可控照片级观测：几何/语义跟控制图走，外观跟 prompt 走——用来补 Sim2Real 合成数据，而不是当物理引擎。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| W2W | World-to-World | 视频条件翻译，区别于 Predict 的 T2W/I2W/V2W |
| ControlNet | Control Network | 冻结基座、另训控制支再加回主去噪 |
| Vis | Visual blur control | 双边模糊支：保颜色与粗构图 |
| HDMap | High-Definition Map | AV 支：车道、标线、3D box |
| SDG | Synthetic Data Generation | Transfer 的主工程用法 |
| PAI-Bench | Physical AI Bench | [PAI-Bench-C](./paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md) 条件生成控制遵循 / 画质榜 |

## 为什么重要

- **解析仿真解决不了「看起来假」：** [Newton](./newton-physics.md) / [Omniverse](./nvidia-omniverse.md) 给几何与接触；Transfer 吃仿真 RGB + depth/seg，吐策略能吃的照片级视频。产品 FAQ 把这条写成官方分工。
- **控制可分区：** Transfer1 的时空权重图让前景跟 Edge/Vis（保机器人外形），背景跟 Depth/Seg（换厨房材质）。这是比「整段 style transfer」更接近机器人数据增广的接口。
- **体积在缩小：** [Predict2.5 论文](./paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md) 报 Transfer2.5-2B 比 Transfer1-7B **小 3.5×** 仍提高控制遵循与画质。
- **配方已经按任务拆好：** [Cookbook](./cosmos-cookbook.md) 有 CARLA、X-Mobility、农机、仓库、GR00T-Mimic；不要从零猜 JSON。

## 核心原理

### 两代对照

| 代 | 基座 | 规模 | 控制接口 | 工程入口 |
|----|------|------|----------|----------|
| **Transfer1（2025-03）** | Predict1 Video2World DiT | 7B；AV Sample；4K Upscaler | 分模态支 + \(\mathbf{w}\in\mathbb{R}^{N\times X\times Y\times T}\) | [cosmos-transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1) |
| **Transfer2.5（2025-10）** | Predict2.5 flow | **2B**；auto 多视角；robot-multiview | JSON `controlnet_specs`；可当场算 depth/seg | [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) |
| **Cosmos 3（2026-06）** | 全模态 MoT | 4B / 16B / 64B | 统一 Generator；Edge **不支持** V2V transfer | [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) |

**核对说明：** 本页「Transfer1」一栏及下文机制/数字已对照 arXiv:2503.14492v2 全文（2026-11-07）；**Transfer2.5 与 Cosmos 3 两栏沿用既有归档（官方文档 / 仓库 / Predict2.5 摘录），本次未回到原文复核**。

机制细节与 TransferBench 数字见 [Transfer1 论文](./paper-cosmos-transfer1.md)；2.5 代 PAI-Bench-C 见 [PAI-Bench 基准](./paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md) 与 [Predict2.5 / Transfer2.5](./paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)。

### Transfer1 论文的关键机制与数字

- **结构：** 在 Cosmos-Predict1 DiT 上每种模态加一条 ControlNet 控制支（3 个 transformer 块，权重取自基座对应块，输出经**零初始化线性层**加回主支，训练时基座冻结）。各支**单独训练、推理时融合**：省显存、各模态可用不同数据、推理时可增删模态。融合时第 $j$ 块的输出乘以时空权重 $w_i\in\mathbb R^{X\times Y\times T}$ 再加回；若各模态权重在某处之和大于 1 则归一化。控制图可手工设计、按启发式规则推导，或由神经模块预测。
- **训练：** 每个控制支用 1024 张 H100 训练 2–4 周（依模态）；一次推理出 5 秒 1280×704、24 fps 视频（约 5.6 万 token）。模态预处理：Vis=双边模糊，Edge=Canny（阈值随机增强），Depth=DepthAnythingV2 归一化，Seg=GroundingDINO + SAM2（颜色随机化，不带语义）；AV 版另有 HDMap（含 3D 框）与 LiDAR（10 Hz 扫描插值到 30 fps 并投影）。AV 数据 RDS-HQ：6.5 万段 20 秒环视片段（约 360 小时）+ 10 Hz LiDAR。4K Upscaler：Real-ESRGAN 式退化训练，推理时 3×3 重叠网格分块去噪取平均。
- **TransferBench（600 条：AgiBot World / OpenDV / Ego-Exo4D 各 200）Table 1：** Vis 控制的 Blur SSIM 0.96，Edge 控制的 Edge F1 0.28（该指标要求像素级细边对齐，所有方法都偏低）；均匀加权全模态（每模态 0.25）Quality Score 8.54，高于所有单模态（5.48–6.51）；去掉 Vis / Edge 使多样性 LPIPS 升到 0.37 / 0.31，去掉 Depth / Seg 降到 0.25 / 0.23，说明稠密结构（Vis/Edge）高保真低多样、稀疏结构（Depth/Seg）相反。
- **时空控制图（SalientObject：VLM 判前景/背景）：** 前景 Vis+Edge 权重各 0.5、背景 Depth+Seg 各 0.5 时，前景 Blur SSIM 0.81、背景 Depth si-RMSE 0.52；把前背景模态对调，多样性 LPIPS 前景 0.01→0.12、背景 0.03→0.33，质量持平。前景 Vis 权重与前景 Blur SSIM 的 Pearson 相关 0.93；背景 Depth 权重与 si-RMSE 相关 −0.92。
- **机器人 Sim2Real（Isaac Lab 厨房，20 场景 × 6 prompt = 120 视频，Table 3）：** Setting 1（前景 Edge+Vis、背景 Seg，权重均为 1）Quality 9.57、FG Mask mIoU 0.61；Setting 2（前景仅 Edge、背景 Seg）Quality 10.42、FG mIoU 0.63、Diversity 0.58；单模态 Seg 的 Diversity 最高（0.60）但前景机器人易出伪影、Blur SSIM 仅 0.47。
- **AV（Table 4）：** HDMap 3D-Bbox mAP 41.89 / Lane mIoU 50.37 / 重投影误差 9.46；LiDAR 46.50 / 48.19 / 8.60；两者融合（$w_{map}=0.3,w_{lidar}=0.7$）44.66 / 51.55 / 8.67——LiDAR 保障碍物位置、HDMap 保车道，融合车道最好。用 Omniverse Sensor RTX 模拟的 LiDAR 也能驱动（尽管训练用真实 LiDAR）。
- **实时（Table 5，B200）：** 1 卡端到端 141.7 s → 64 卡 4.2 s（扩散部分 141.0 s → 3.5 s，约 40× 加速）；策略为非注意力层纯数据并行 + 注意力层按头并行（all-to-all），正负 prompt 的 CFG 分到两组 GPU。
- **Prompt upsampler：** 微调 Pixtral-12B，输入条件视频 + 短 prompt，每种模态 100 万视频、训练一个 epoch（附录 A）。

### 流程总览

```mermaid
flowchart LR
  subgraph src [输入世界]
    SIM[Isaac / CARLA / 真机 RGB]
    MAP[depth / edge / seg / vis<br/>或 HDMap / LiDAR]
  end
  subgraph xfer [Transfer]
    SPEC[JSON 控制规格<br/>weight + 可选 mask]
    NET[多 ControlNet 支]
    TXT[文本 prompt]
    SPEC --> NET
    MAP --> NET
    TXT --> NET
  end
  SIM --> MAP
  NET --> OUT[照片级视频]
  OUT --> POL[VLA / 感知 / 评测]
```

**读法：** 控制图钉住「什么在哪、怎么动」；prompt 改「像什么时候、什么材质」。把输出当守恒律仿真会误判。

### 控制模态怎么选

| 模态 | 保住什么 | 放开什么 | 典型用法 |
|------|----------|----------|----------|
| **Vis（模糊）** | 颜色、粗构图 | 纹理锐度 | CG→真、保机器人涂装 |
| **Edge** | 轮廓 | 颜色与材质 | 换外观、保外形 |
| **Depth** | 3D 几何 | 语义与纹理 | 保桌面/车道布局 |
| **Seg** | 物体布局 | 实例外观 | 换背景、加杂物 |
| **HDMap / LiDAR** | 路网与 3D 障碍 | 天气与光照 | AV Sample |

密结构（Vis/Edge）对齐高、多样性低；疏结构（Depth/Seg）相反。多控均匀加权在 Transfer1 TransferBench 上 Quality Score **8.54**，高于任一单模态。

## 工程实践

| 目标 | 从哪进 | 备注 |
|------|--------|------|
| 新产品 / 全模态 | [Cosmos 3](./cosmos-3.md) | 官方 2026-06 起停更 Transfer 仓；Edge 不做 V2V |
| 复现 2.5 / 旧配方 | `cosmos-transfer2.5` + [Cookbook](./cosmos-cookbook.md) | `examples/inference.py -i spec.json`；多卡 `torchrun` |
| 复现 1.0 论文 / 4K / AV Sample | `cosmos-transfer1` | 7B；Edge Distilled 1 步；Llama Guard 3 过滤 |
| 仿真→照片级 | Cookbook：CARLA、Warehouse、X-Mobility、农机 | 先保证仿真 depth/seg 对齐，再调 weight |
| 真机多样性 | Cookbook：Weather、Real-World Manipulation | 同一段 RGB 换 prompt，勿改控制图 |
| ROBOTIS AI Worker | [Humanoid Skills 技术地图](../overview/robotis-humanoid-skills-nvidia-stack-technology-map.md) | **268** 真机 + **300** Transfer 2.5 增广 → GR00T 1.7；缓解阴影误检 |

开源结论（2026-09-05 项目页核查）：**Transfer1 / Transfer2.5 代码与权重已开源**（Apache-2.0 + NVIDIA Open Model License）。两仓 README 均写 **有限维护**。HF 仓多为门控。

### Transfer2.5 可跑入口

官方推理在 `docs/inference.md`。单卡 2B 约 **65.4 GB**；720p / 16 fps / 93 帧 segmentation：B200 扩散约 92 s。蒸馏 Edge 4 step，单卡相对 base 约 **7.4–7.8×**。

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Setup as docs/setup.md
    participant HF as HF Transfer2.5-2B
    participant Inf as examples/inference.py
    participant Ctrl as depth/edge/seg/vis
    participant DiT as Predict2.5 + ControlNet
    participant Out as MP4
    Dev->>Setup: uv + CUDA extra
    Dev->>HF: 拉 general / auto / distilled
    Dev->>Inf: JSON spec + prompt + 视频
    Inf->>Ctrl: 读 control_path 或当场计算
    Ctrl->>DiT: 加权多控
    DiT->>Out: 720p chunk / 滑动窗
    Out-->>Dev: 合成演示
```

后训练：`docs/post-training_singleview.md`（edge/depth/seg/blur）与 `post-training_auto_multiview.md`。农机深度条件后训练见 Cookbook 2026-04-21 配方。

## 局限与风险

- **论文中的边界：** 仅在 TransferBench 与两个案例上评测；机器人案例只有 20 个仿真场景，且质量/多样性指标是自动指标，没有下游策略训练收益的对照；Edge F1 这类指标本身严苛。论文没有给出用 Transfer 数据训练策略后真机成功率提升的实验。

- **像素世界 ≠ 物理引擎：** 控制图对齐高也不保证接触力、摩擦、关节限位正确。策略仍要真机或解析仿真验收。
- **仓已停更：** 从 Transfer2.5 起步会接到「请迁 Cosmos 3」。Cosmos 3 Edge **明确不支持** video-to-video transfer——需要 V2V 时仍只能停在 2.5 或等 3.x 补口。
- **算力：** Transfer1-7B 单卡生成 5 s 720p 约 142 s；实时要 64×B200。Transfer2.5-2B 单卡仍要 65 GB 级。
- **安全过滤：** Transfer1 用 Llama Guard 3（独立许可）；关掉 guardrail 只为测速，不能当产品默认。

## 关联页面

- [Cosmos-Transfer1 论文](./paper-cosmos-transfer1.md) — 自适应多控与 TransferBench
- [Predict2.5 / Transfer2.5 论文](./paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)
- [Cosmos Cookbook](./cosmos-cookbook.md) — 可运行配方
- [NVIDIA Cosmos 平台](./nvidia-cosmos.md)
- [Cosmos 3](./cosmos-3.md) — 当前母栈
- [Newton Physics](./newton-physics.md)
- [NVIDIA Omniverse](./nvidia-omniverse.md)
- [NVIDIA SO-101 Sim2Real 动手课](./nvidia-so101-sim2real-lab-workflow.md) — Strategy 3 增广
- [Generative World Models](../methods/generative-world-models.md)
- [Sim2Real](../concepts/sim2real.md)
- [Video-as-Simulation](../concepts/video-as-simulation.md)
- [Manipulation](../tasks/manipulation.md)

## 公司路线日期口径

Transfer1 的 arXiv:2503.14492 v1 提交于 2025-03-18；Transfer2.5 等后续版本另计。 [日期证据](https://arxiv.org/abs/2503.14492)。详见[本轮日期核查](../../sources/sites/company-roadmap-date-audit-2026-10-08.md)；版本事件与原始产品首发分别记录。

## 参考来源

- **原文通读：** [Transfer1 论文归档](../../sources/papers/cosmos_transfer1_arxiv_2503_14492.md)（arXiv v2 全文，2026-11-07）
- [公司路线日期核查](../../sources/sites/company-roadmap-date-audit-2026-10-08.md)

- [Transfer1 项目页](../../sources/sites/cosmos-transfer1-project.md)
- [cosmos-transfer1 仓库](../../sources/repos/nvidia_cosmos_transfer1.md)
- [Transfer1 论文摘录](../../sources/papers/cosmos_transfer1_arxiv_2503_14492.md)
- [Transfer2.5 官方文档](../../sources/sites/cosmos-transfer25-docs.md)
- [cosmos-transfer2.5 仓库](../../sources/repos/nvidia_cosmos_transfer25.md)
- [Cosmos Cookbook 站点](../../sources/sites/cosmos-cookbook.md)
- [Predict2.5 论文摘录](../../sources/papers/cosmos_predict25_arxiv_2511_00062.md)
- [human five · Humanoid Skills（ROBOTIS Worker Cosmos 案例）](../../sources/blogs/wechat_human_five_robotis_humanoid_skills_nvidia_stack_2026-09-28.md)

## 推荐继续阅读

- [Cosmos Cookbook · Control Modalities](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html)
- [GitHub: cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [GitHub: cosmos-transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1)
- [arXiv:2503.14492](https://arxiv.org/abs/2503.14492)
- [NVIDIA Docs: Transfer2.5](https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html)
