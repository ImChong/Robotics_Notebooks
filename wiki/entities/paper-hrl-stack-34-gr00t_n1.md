---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, loco-manipulation, loco-manip-161-survey, nvidia, vla]
status: complete
updated: 2026-11-13
arxiv: "2503.14734"
venue: "2025 · arXiv"
code: https://github.com/NVIDIA/Isaac-GR00T
summary: "GR00T N1（arXiv:2503.14734）是 NVIDIA 开源的人形通才 VLA：Eagle-2 VLM（System 2）+ DiT flow-matching 动作头（System 1），以数据金字塔联合真机/仿真/人类视频预训练；GR00T-N1-2B 在 GR-1 真机双臂操作达 76.8% 平均成功率且 10% 数据仍 42.6%。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-09-vla-world-models.md
  - ../concepts/foundation-policy.md
  - ../methods/vla.md
  - ../methods/diffusion-policy.md
  - ../entities/gr00t-wholebodycontrol.md
  - ../entities/isaac-gr00t.md
  - ./paper-gr00t-n1-5.md
  - ../../roadmap/depth-robotics-diffusion-dit-flow.md
  - ./paper-deed.md
  - ./paper-data-pyramid-embodied-manipulation.md
  - ./paper-pi-r2.md
  - ./paper-fsd-vln.md
original:
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
sources:
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md
  - ../../sources/repos/isaac_gr00t.md
  - ../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/loco_manip_161_survey_148_gr00t-n1.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
---

# GR00T N1

**GR00T N1**（*An Open Foundation Model for Generalist Humanoid Robots*，[arXiv:2503.14734](https://arxiv.org/abs/2503.14734)，[NVIDIA 白皮书](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)，[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)）是 NVIDIA GEAR 提出的 **开源人形通才基础模型**，面向语言条件的跨具身操作（单臂、双臂、人形）。

> **工程平台：** [Isaac GR00T](../entities/isaac-gr00t.md) — N1.7 GA 代码、LeRobot 数据管线、后训练与部署入口（[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)）。  
> **深读页：** [gr00t-wholebodycontrol](../entities/gr00t-wholebodycontrol.md) — N1.5/N1.6/N1.7 工程栈、WBC 与部署细节见链接页；本页以 N1 论文机制与评测为主。

## 一句话定义

GR00T N1 是双系统 **Vision-Language-Action（VLA）** 模型：**System 2** 用 Eagle-2 VLM 理解图像与语言指令，**System 1** 用 DiT flow-matching 以 120Hz 生成 action chunk；在 **数据金字塔**（人类视频 → 仿真/神经合成 → 真机遥操作）上端到端联合预训练，公开 **GR00T-N1-2B** checkpoint 并在 Fourier GR-1 真机验证高数据效率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型骨干 |
| DiT | Diffusion Transformer | 以 Transformer 实现的扩散/流匹配生成模块 |
| IDM | Inverse Dynamics Model | 由观测序列反推动作标签的逆动力学模型 |
| LAPA | Latent Action Pretraining from Videos | 从视频学 latent action 的预训练范式（论文中作独立 embodiment） |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#34/42）。
- 在 [人形 Loco-Manip 161 篇技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md) 中属于 **09 人形 VLA、世界模型与通用操作**（#148/161）；本页为合并后的 **单一 canonical 实体**。
- 论文把「人形 foundation model」落到可复现开源栈：**GR00T-N1-2B**（总参数 2.2B，其中 VLM 1.34B）、训练数据与 RoboCasa / DexMG / GR-1 仿真 benchmark 一并公开。
- 核心工程贡献不仅是 VLA 口号，而是 **跨具身动作接口**（per-embodiment 状态/动作 MLP、action chunk、latent/IDM 伪动作）与 **数据金字塔共训练**，缓解单一人形硬件数据孤岛问题。

## 核心信息

| 字段 | 内容 |
|------|------|
| 原文题目 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots |
| arXiv | [2503.14734](https://arxiv.org/abs/2503.14734)（[HTML v2](https://arxiv.org/html/2503.14734v2) · [PDF](https://arxiv.org/pdf/2503.14734)） |
| 官方白皮书 | [GR00T_1_Whitepaper.pdf](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)（2025-03-18） |
| 机构 | NVIDIA（GEAR；Linxi Fan、Yuke Zhu 等） |
| 公开模型 | GR00T-N1-2B |
| 代码 | <https://github.com/NVIDIA/Isaac-GR00T> |
| 数据集 | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim> |
| 真机平台 | Fourier GR-1（语言条件双臂桌面操作） |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 34/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在人形 Loco-Manip 161 篇中

| 字段 | 内容 |
|------|------|
| 编号 | 148/161 |
| 分组 | 09 人形 VLA、世界模型与通用操作 |
| 分类 hub | [loco-manip-161-category-09-vla-world-models](../overview/loco-manip-161-category-09-vla-world-models.md) |
| 索引来源 | [具身智能研究室 · 161 篇人形 Loco-Manip 长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) |

## 核心机制（归纳）

### 1）双系统 VLA 架构

受 Kahneman 双过程理论启发，GR00T N1 将 **推理** 与 **运动生成** 拆为两个 Transformer 模块并端到端联合训练：

| 模块 | 角色 | 关键规格（论文） |
|------|------|------------------|
| **System 2** | Eagle-2 VLM：理解任务语言与视觉场景 | SmolLM2 + SigLIP-2；224×224 图像 → 每帧 64 token；取 **第 12 层** LLM 隐状态（较末层更快且下游成功率更高）；L40 上约 **10Hz** |
| **System 1** | DiT flow-matching：生成闭环电机命令 | cross-attend VLM token；**action chunk** $H=16$；推理 **$K=4$** 步去噪；L40 bf16 采样 16 步动作约 **63.9ms**（≈**120Hz** 闭环） |

每个 robot embodiment 有独立的 **State Encoder / Action Encoder / Action Decoder** MLP，把不同维度的本体状态与动作投影到共享 DiT 隐空间。

### 2）数据金字塔与共训练

```mermaid
flowchart TB
  subgraph pyramid [数据金字塔（自下而上：数量↓、具身特异性↑）]
    base["底层：Web 数据 + 人类 egocentric 视频\n（Ego4D、Ego-Exo4D、HOI4D 等）"]
    mid["中层：合成数据\n仿真 DexMimicGen + image-to-video 神经轨迹"]
    top["顶层：真机遥操作\nGR-1 内部集 + Open X-Embodiment + AgiBot-Alpha"]
    base --> mid --> top
  end
  model["GR00T N1 预训练 / 后训练\n异构 batch 共采样"]
  pyramid --> model
```

- **无动作标签视频：** 训练 VQ-VAE **latent action**（编码器吃 $x_t$ 与 $x_{t+H}$，解码器重建 $x_{t+H}$；训练后取编码器的量化前连续嵌入作标签，当作独立的「LAPA」embodiment）与 **IDM 伪动作**（沿用 DiT + SigLIP-2、flow-matching 训练、每个 embodiment 单独训 3 万或 6 万步），把人类/神经视频当作额外 embodiment 纳入 flow-matching。人类视频来源：Ego4D、Ego-Exo4D、Assembly-101、EPIC-KITCHENS、HOI4D、HoloAssist、RH20T-Human；真机来源另含 OXE 子集（RT-1、Bridge-v2、Language Table、DROID、MUTEX、RoboSet、Plex）与 AgiBot-Alpha（14 万条）。预训练总计约 **8375.7 小时 / 5.93 亿帧**（真机 3288.8 h、人类视频 2517.0 h、仿真 1742.6 h、神经 827.3 h，Table 7）。
- **神经轨迹：** 用 LoRA 微调 WAN2.1-I2V-14B（3000 条 480P×81 帧带语言标注的 GR-1 样本，100 epoch），在 88 小时遥操作基础上生成约 **827 小时**（约 30 万条，≈10×）反事实语言条件轨迹；先由多模态 LLM 枚举物理可行的「把 A 从 B 移到 C」指令，再抽 8 帧让 MLLM 判官过滤并重新标注；生成成本约 10.5 万 L40 GPU 小时。后训练可与真机轨迹 **1:1** 共采样。
- **仿真轨迹：** DexMimicGen（把人类 demo 切成以物体为中心的子任务片段，按新物体位姿变换、插值拼接并回放，仅保留成功轨迹）在 RoboCasa 框架下，从 Leap Motion 采集的数十份源 demo 出发，为 54 组（源,目标）容器组合各生成 1 万条，预训练共 **54 万条**；预训练 + 后训练合计 **78 万条 ≈ 6500 小时人类演示等价，仅用 11 小时生成**（论文 §2.2/§3.2）。
- **类目级系统化：** 本节为 **单模型配方视角** 的三层金字塔；[具身数据金字塔综述](./paper-data-pyramid-embodied-manipulation.md)（arXiv:2607.24744）将其与 Motus 等模型专属金字塔统一为 **五层类目级坐标系**（真机/UMI/Ego-Exo/仿真/通用 × 六维属性），作跨模型数据选型对照阅读。

### 3）预训练与后训练

- **预训练：** 跨真机、仿真、人类视频与神经轨迹混合采样；人类视频只用 latent action，真机数据同时用真值动作与 latent action，神经轨迹用 latent + IDM 预测动作作目标；最多 **1024×H100**，GR00T-N1-2B 预训练约 **5 万 H100 GPU 小时**（OSMO 编排 + Ray 容错）。损失除式(1)的 flow-matching 外，还加辅助**目标检测损失**：OWL-v2 标注指令目标的框中心，在 VLM 末层 token 后接线性层回归归一化中心坐标，$L=L_{fm}+L_{det}$。
- **后训练：** 按单一具身微调；默认冻结 VLM **语言** 部分；低数据场景可仅调 adapter（state/action 编解码 + DiT）或连同视觉编码器（A6000 上 batch 最高 200 vs 16）。

### 4）流匹配目标与推理（论文式 (1)）

加噪动作块 $A_t^\tau=\tau A_t+(1-\tau)\epsilon$，模型 $V_\theta(\varphi_t,A_t^\tau,q_t)$ 拟合去噪向量场 $\epsilon-A_t$：$\mathcal L_{fm}=\mathbb E_\tau\|V_\theta-(\epsilon-A_t)\|^2$，$\tau$ 采样自 $\mathrm{Beta}((s-\tau)/s;1.5,1)$，$s=0.999$；推理从 $A_t^0\sim\mathcal N(0,I)$ 出发做前向 Euler：$A_t^{\tau+1/K}=A_t^\tau+\frac1K V_\theta(\cdot)$，$K=4$。DiT 以 adaLN 注入去噪步，自注意力处理加噪动作 token + 状态 token，交叉注意力读取 VLM 特征；与 π₀ 的 mixture-of-experts 桥接不同，GR00T 只用简单交叉注意力，便于替换 VLM 或动作模块。

## 实验与评测

### 预训练泛化（GR-1 真机，预训练 checkpoint）

| 任务 | 成功率 |
|------|--------|
| 双手递送后放置（物体故意偏左，需左手抓取再交右手） | **76.6%**（11.5/15） |
| 未见物体放入未见容器 | **73.3%**（11/15） |

### 仿真后训练（每任务 100 条 demo，平均成功率）

| 方法 | RoboCasa | DexMG | GR-1 Tabletop | **平均** |
|------|----------|-------|---------------|----------|
| BC-Transformer | 26.3% | 53.9% | 16.1% | 26.4% |
| Diffusion Policy | 25.6% | 56.1% | 32.7% | 33.4% |
| **GR00T-N1-2B** | **32.1%** | **66.5%** | **50.0%** | **45.0%** |

### 真机后训练（GR-1，四类桌面任务平均）

| 方法 | 数据量 | 平均成功率 |
|------|--------|------------|
| Diffusion Policy | 10% | 10.2% |
| Diffusion Policy | 全量 | 46.4% |
| **GR00T-N1-2B** | **10%** | **42.6%** |
| **GR00T-N1-2B** | **全量** | **76.8%** |

分任务类别（Table 3，Full Data / 10% Data）：Pick-and-Place 82.0% / 35.0%（DP 36.0% / 3.0%）；Articulated 70.9% / 62.0%（DP 38.6% / 14.3%）；Industrial 70.0% / 31.0%（DP 61.0% / 6.7%）；Coordination 82.5% / 50.0%（DP 62.5% / 27.5%）。10% 数据 GR00T N1 仅比 DP 全量低 3.8 个百分点。

### 数据量扫描与神经轨迹消融

- 仿真每任务 30 / 100 / 300 条 demo（Fig. 10 / Table 4）：GR00T-N1-2B 平均 RoboCasa 17.4 / 32.1 / 49.6，DexMG 29.6 / 58.5 / 74.2，GR-1 43.2 / 50.0 / 49.3；DP 分别为 14.7 / 25.6 / 43.2、23.7 / 46.9 / 68.4、21.3 / 32.7 / 40.4。论文承认：微调数据足够多时预训练的优势会减弱（GR-1 300 条时两者差距缩到约 9 点）。
- 神经轨迹共训练（RoboCasa 每任务 3k 条神经轨迹；真机每任务 100 条）：RoboCasa 30 / 100 / 300 条区分别 **+4.2% / +8.8% / +6.8%**，GR-1 真机 8 个任务平均 **+5.8%**；LAPA 标签在最低数据量（30）略优，数据更多时 IDM 标签优势拉大，因此真机只用 IDM 标签。
- 评测协议：仿真 100 次试验、取最后 5 个 checkpoint（每 500 步一个）的最大值；真机每任务 10 次（Pack Machinery 仅 5 次）并采用部分得分（抓对但未放入记 0.5）；默认全局 batch 1024、训练 6 万步，DexMG 用 batch 128。

### 局限（论文 §4.6）

当前 N1 主要覆盖 **短视界桌面操作**；长视界 loco-manipulation、更强空间推理 VLM、物理一致合成数据与架构仍属后续方向（与 [Foundation Policy](../concepts/foundation-policy.md) 中 GR00T N1.5+ 演进线对照阅读）。

## 工程实践

| 项 | 论文给出的做法 / 数值 |
|----|----------------------|
| 数据格式 | 扩展 LeRobot：parquet 表格 + MP4 + JSON 元数据；强制 `meta/modality.json` 描述状态/动作各字段语义、数据类型、范围、旋转表示，支持多种标注（附录 E.1） |
| 动作空间统一 | 末端旋转状态用 6D 表示，旋转动作用 axis-angle；关节与末端位置做 min-max 归一化；向量顺序为末端旋转、末端位置、夹爪，先左臂后右臂（附录 E.2） |
| 预训练超参 | AdamW，lr 1e-4，cosine，warmup 0.05，weight decay 1e-5；batch 16,384、20 万步（Table 6） |
| 后训练超参 | batch 128 或 1024、2 万–6 万步；冻结 VLM 语言部分（文本 tokenizer），其余可训（Table 6 / §2.3） |
| 低算力微调 | 单张 A6000：仅调适配层（状态/动作编解码 + DiT）batch 可到 200；连视觉编码器一起调最高 16（§2.3） |
| 推理 | L40 + bf16 采样 16 步动作块 63.9 ms；VLM 约 10 Hz、动作头闭环约 120 Hz |
| 真机数据采集 | VIVE Ultimate Tracker 采腕部位姿 + Manus/Xsens 手套采手指，也试过 Apple Vision Pro、Leap Motion；IK 重定向，20 Hz；每任务后训练数据 15 分钟–3 小时并过滤低质量轨迹 |

官方代码侧的运行入口（Isaac-GR00T 仓库当前版本）见下节时序图；论文与代码在模型版本上并不等同（仓库已演进到 N1.5/N1.6/N1.7），复现 N1 论文数字应以论文 checkpoint 与协议为准（推测：具体复现命令以仓库对应 tag 的 README 为准，未在本次通读论文时核实）。

## 源码运行时序图

官方参考实现 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：微调 `gr00t/experiment/launch_finetune.py`；推理 `scripts/deployment/standalone_inference_script.py` 或 `gr00t/eval/run_gr00t_server.py`；开环评估 `gr00t/eval/open_loop_eval.py`。数据为 GR00T LeRobot 格式 + `modality.json`。一次完整运行如下：

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant DS as LeRobot 演示<br/>modality.json
    participant FT as launch_finetune.py
    participant VLA as GR00T VLA<br/>+ diffusion action head
    participant SRV as run_gr00t_server.py
    participant CLI as 机器人 / 仿真客户端
    U->>DS: 准备 embodiment 标签与模态配置
    U->>FT: --base-model-path nvidia/GR00T-…
    FT->>VLA: 微调
    U->>SRV: 启动 PolicyServer
    loop 闭环
        CLI->>SRV: 图像 / 状态 / 语言
        SRV->>VLA: 前向
        SRV-->>CLI: 动作 chunk
        CLI->>CLI: 执行并反馈
    end
    U->>FT: （可选）open_loop_eval.py 评测
```

- **N1 论文机制 vs N1.5+ 工程**：本时序对应当前 Isaac-GR00T 发布栈；版本号以模型卡为准。
- **全身低层**常再接 [SONIC / GR00T-WholeBodyControl](../methods/sonic-motion-tracking.md)。

## 结论

**通才人形 VLA 的工程杠杆是数据金字塔共训练 + 跨具身动作接口；System 2 理解与 System 1 的 ~120 Hz action chunk 解耦，不自动替代底层 WBC。**

1. **真机数据效率** — GR-1 四类桌面任务：全量 **76.8%**；仅 **10%** 数据仍 **42.6%**，接近 Diffusion Policy 全量 **46.4%**（DP 10% 仅 10.2%）。
2. **仿真后训练** — 每任务 100 demo 平均成功率 **45.0%**（RoboCasa/DexMG/GR-1 Tabletop），高于 DP **33.4%** 与 BC-Transformer **26.4%**。
3. **预训练已有泛化** — 双手递送后放置 **76.6%**；未见物体放入未见容器 **73.3%**（预训练 checkpoint）。
4. **合成数据怎么用** — 神经轨迹约 **10×** 反事实扩增，后训练可与真机 **1:1** 共采样；低数据区约额外 **+4%～+9%**。
5. **实现细节** — VLM 取第 **12** 层隐状态（较末层更快且下游更高）；DiT flow-matching \(H=16\)、推理 \(K=4\)；每 embodiment 独立 state/action MLP。
6. **后训练风险** — 仅右手数据微调可能洗掉预训练双手递送等能力；当前主覆盖短视界桌面操作，长视界 loco-manip 仍属后续。

## 与其他工作对比

| 项目 | 定位区别 |
|---|---|
| [Diffusion Policy](../methods/diffusion-policy.md) | 单任务从零训练的生成式 BC；本文作主要基线，GR00T N1 以预训练 + 后训练在 10% 数据下 42.6% 接近 DP 全量 46.4% |
| [π₀](./paper-pi0.md) | 同为 VLM + flow-matching 动作专家；π₀ 用 mixture-of-experts 桥接，GR00T N1 只用简单交叉注意力，便于替换 VLM 或动作模块（论文未做正面对比） |
| [GR00T N1.5](./paper-gr00t-n1-5.md) | 后继版本：冻结 Eagle VLM + FLARE 未来 latent 对齐，语言跟随与少样本 post-train 优于 N1 |
| [具身数据金字塔综述](./paper-data-pyramid-embodied-manipulation.md) | 本文是单模型配方的三层金字塔；综述将其统一为五层类目级坐标系，作跨模型数据选型参照 |

## 局限与风险

**论文自述（§4.6）：** 当前主要覆盖短视界桌面操作；长视界 loco-manipulation 需要更强硬件、架构与训练语料；合成数据在「多样性、反事实、遵守物理」上仍有挑战；期待更强的 VLM 以提升空间推理与语言理解。**论文报告的边界：** 神经轨迹共训练的结论是「初步结果」，仅在 RoboCasa 与两类真机任务上量化；真机评测是 GR-1 上单一硬件、每任务 10 次试验，统计置信区间较宽；对比基线只有 BC-Transformer 与 Diffusion Policy，没有与其他 VLA（如 π₀、OpenVLA）的正面对比。

### 常见误区

1. **VLA 条目解决接口与预测，不自动替代底层 WBC** — 真机执行仍依赖 Fourier GR-1 全身 IK / 控制栈；高层策略与低层运控需分层验证（见 [gr00t-wholebodycontrol](../entities/gr00t-wholebodycontrol.md)）。
2. **后训练会洗掉部分预训练行为** — 论文指出仅右手数据的 post-training 可能丢失预训练学到的双手递送等能力。
3. **161 篇策展坐标 ≠ 论文全部指标** — 仿真/真机数字以 arXiv / 白皮书为准；策展公众号提供地图定位，不替代原文 benchmark。

## 关联页面

- 概念层：[foundation-policy.md](../concepts/foundation-policy.md)、[vla.md](../methods/vla.md)
- 方法对照：[diffusion-policy.md](../methods/diffusion-policy.md)
- **长程上下文 / 部署后学习：** [RoboTTT](./paper-robottt-test-time-training-vla-context.md) — 在 **GR00T N1.7** 上内嵌 TTT fast-weight 层，把 visuomotor 上下文扩到 8K 步（项目页，2026）
- **零售真机后训练配方（未开源）：** [DEED](./paper-deed.md) — G1-Edu + GR00T N1.6 薯片补货；Data-Efficient SFT + 文本 advantage 前缀 RECAP
- **对象中心可验证闭环（未开源）：** [POT-VLA](./paper-pot-vla.md) — 同 **GR00T-N1.7** 匹配对照下，持久 3D 对象 token + 几何谓词把 G1 八类任务 **39/80→71/80**
- **空中 VLN 双系统迁移（未开源）：** [FSD-VLN](./paper-fsd-vln.md) — 用 **GR00T N1** 骨干做 UAV 语言导航：冻结 VLM、只训 DiT 决策头；仅仿真
- 工程平台：[isaac-gr00t.md](../entities/isaac-gr00t.md)
- 后继版本：[GR00T N1.5](./paper-gr00t-n1-5.md)；合成数据来源见 [NVIDIA Cosmos](./nvidia-cosmos.md)
- 工程深读：[gr00t-wholebodycontrol.md](../entities/gr00t-wholebodycontrol.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- Loco-Manip 161 篇：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)

## 参考来源

- **原文通读：** [gr00t_n1_arxiv_2503_14734.md](../../sources/papers/gr00t_n1_arxiv_2503_14734.md)（arXiv v2 全文含附录，2026-11-07）
- [isaac_gr00t.md](../../sources/repos/isaac_gr00t.md) — Isaac-GR00T 仓库与 N1.7 GA 工程摘录
- [gr00t_n1_arxiv_2503_14734.md](../../sources/papers/gr00t_n1_arxiv_2503_14734.md) — arXiv:2503.14734 + NVIDIA 白皮书策展摘录
- [humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md](../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md) — 42 篇栈策展摘录
- [loco_manip_161_survey_148_gr00t-n1.md](../../sources/papers/loco_manip_161_survey_148_gr00t-n1.md) — Loco-Manip 161 #148 策展摘录
- arXiv：<https://arxiv.org/abs/2503.14734>
- 白皮书：<https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf>
- 代码：<https://github.com/NVIDIA/Isaac-GR00T>
- 数据集：<https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim>

## 推荐继续阅读

- [机器人论文阅读笔记：GR00T N1 Humanoid Foundation Model](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/GR00T_N1_Humanoid_Foundation_Model/GR00T_N1_Humanoid_Foundation_Model.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [161 篇 Loco-Manip（微信公众号）](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A)
- [GR00T-WholeBodyControl 仓库实体](./gr00t-wholebodycontrol.md)
- [πR²](./paper-pi-r2.md) — 在 GR00T-N1.7 上做反应式实时 flow 闭环部署（约 25 Hz；代码已开）
