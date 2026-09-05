---
type: entity
tags: [paper, manipulation, real2sim, sim2real, scene-generation, digital-twin, digital-cousin, policy-evaluation, nvidia, gear, droid, isaac-lab, omnigibson, 3dgs, vla]
status: complete
updated: 2026-09-05
arxiv: "2606.28276"
code: https://github.com/NVlabs/SimFoundry
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../concepts/sim2real.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
  - ../methods/vla.md
  - ../methods/crisp-real2sim.md
  - ./nvidia-gear-lab.md
  - ./isaac-gym-isaac-lab.md
  - ./behavior-1k.md
  - ./physx-omni.md
  - ./genesis-world-10.md
  - ./paper-hrl-stack-34-gr00t_n1.md
  - ./paper-agentic-real2sim.md
  - ./paper-lucida-r2s.md
sources:
  - ../../sources/papers/simfoundry_arxiv_2606_28276.md
  - ../../sources/sites/nvidia-research-simfoundry.md
  - ../../sources/repos/nvlabs-simfoundry.md
summary: "SimFoundry（arXiv:2606.28276，NVIDIA GEAR）从单段真机视频模块化重建 sim-ready 数字孪生，并自动生成 object/scene/task digital cousins；统一支撑 real-to-sim 策略评测（均值 Pearson 0.911）与 sim-to-real 操作策略训练（DROID/YAM）。官方仓部分开源：A/B 重建与 cousins 可跑，论文级训练/评测未随仓。"
---

# SimFoundry（Modular Real2Sim Scene Generation for Policy Learning and Evaluation）

**SimFoundry** 是 NVIDIA [GEAR Lab](./nvidia-gear-lab.md) 等团队的 **Real2Sim→Sim2Real 闭环系统**（arXiv:2606.28276，2026-06；[官方仓](https://github.com/NVlabs/SimFoundry)）：输入 **单段真实场景 RGB 视频**，全自动产出 **可物理交互的仿真场景**（数字孪生），并沿 **物体实例、空间布局、任务规格** 三轴扩展 **digital cousins**（保 affordance 的语义变体）。同一套环境既用于 **已有真机策略的 real-to-sim 评测**，也用于 **纯仿真演示训练策略并零样本部署真机**。开源默认导出 **[OmniGibson](./behavior-1k.md)** 场景 JSON，而不是论文叙述里的 Isaac Lab。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Real2Sim | Real to Simulation | 从真机观测/视频构造可对齐的仿真场景与资产 |
| Sim2Real | Simulation to Real | 仿真训练策略迁移真机部署 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态操作策略 |
| MMRV | Mean Maximum Rank Violation | 仿真与真机策略排序一致性指标，越小越好 |
| 3DGS | 3D Gaussian Splatting | 高保真背景外观表示，常与前景 mesh 混合 |
| DROID | Distributed Robot Interaction Dataset | 单臂 Franka 操作平台与大规模数据集生态 |
| VLM | Vision-Language Model | 场景理解、物性标注与任务提议的多模态模型 |
| Isaac Lab | NVIDIA Isaac Lab | 论文写的下游导出目标之一；开源默认走 OmniGibson |
| OG | OmniGibson | StanfordVL 仿真运行层；仓内 `s14_og/reconstructed_og_scene.json` |

## 为什么重要

- **闭合「重建—评测—训练」三环：** 许多 Real2Sim 工作止步于网格可视化；SimFoundry 把 **sim-ready 场景**、**程序化数据生成** 与 **与真机相关的 benchmark** 放在同一模块化栈里（论文 Table 1 与 [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) 叙事直接对齐）。
- **可预测的 real-to-sim 排序：** 在 **7 任务 × 5 策略族**（π₀、π₀.₅、GR00T、DreamZero 等）上，仿真成功率与真机 **均值 Pearson r=0.911、MMRV=0.018**，显著优于 **PolaRiS** 类 SOTA 基线——使「先在 sim 里筛 checkpoint」对操作 foundation model 更具工程可信度。
- **Cousins 作为可控域随机化：** **object / scene / task cousins** 不是简单 pose 噪声，而是 **语义与 affordance 保持** 的实例/布局/任务变体；论文报告三者分别带来约 **+17% / +21% / +40%** 平均任务成功率增益，并支撑 **held-out 物体与任务** 泛化（如 π₀.₅-base 在 7 个未见任务上 **0%→29%**）。
- **任务复杂度上探：** 相对既往 real-to-sim 工作，实验覆盖 **多步语言跟随、铰接物体、双手 YAM** 与 **DROID 单臂**，把评测从「原子 pick-place」推到更接近 foundation model 关心的长程操作。

## 流程总览

```mermaid
flowchart TB
  subgraph in [输入]
    vid[单段真机 RGB 视频]
  end
  subgraph ext [Extraction 感知提取]
    rgbd[代表帧 RGB-D + 点云对齐]
    seg[迭代 VLM 检物 + 分割 / inpaint]
    crops[per-object RGB-D crop]
    vid --> rgbd --> seg --> crops
  end
  subgraph gen [Generation 资产生成]
    mesh[2D→3D mesh + 位姿对齐]
    art[铰接物体关节推断]
    phys[碰撞 CoACD + 质量摩擦标注]
    compose[物理沉降 → OmniGibson 场景 JSON]
    crops --> mesh --> art --> phys --> compose
  end
  subgraph aug [Augmentation digital cousins]
    oc[object cousins\n保 affordance 换几何/外观]
    sc[scene cousins\n语义空间布局变体]
    tc[task cousins\nVLM 提议新可行任务]
    compose --> oc
    compose --> sc
    compose --> tc
  end
  subgraph bg [可选背景]
    splat[3D Gaussian Splat\n自动前景擦除或二遍拍摄]
    compose --> splat
  end
  subgraph apps [下游应用]
    eval[Real-to-sim 策略评测\nPearson / MMRV]
    train[Sim-to-real 演示训练\n零样本 / co-train / 多任务]
    oc --> train
    sc --> train
    tc --> train
    splat --> eval
    splat --> train
  end
```

## 核心机制（归纳）

### 模块化 Real2Sim 三阶段

1. **Extraction：** 深度估计 + 地面/世界系对齐；VLM 列举物体后 **逐物体分割并从 RGB-D 中擦除**，直至只剩背景——避免遮挡下漏检，并为每物体保留独立 crop。
2. **Generation：** 图像超分 + **2D→3D 生成 mesh**（开源默认 **Hunyuan3D-2.1**，可选 TRELLIS.2 / Pixal3D）；**FoundationPose** 类模块精化 6D 位姿；橱柜/抽屉等走 **articulation** 分支；碰撞体 + VLM 查询 **质量/摩擦**；物理沉降后导出。论文写 **PyBullet 消解 → Isaac Lab**；**开源仓**编译到 **OmniGibson**（`s14_og/reconstructed_og_scene.json`）。
3. **Augmentation：** 在 **digital twin**（几何与布局严格复刻真场景）上扩展 **digital cousins**：
   - **Object cousins：** 图像空间编辑物体再重生成 mesh（换形状/纹理但保留「可抓/可放」语义）。
   - **Scene cousins：** 用 **OnTop / RightOf** 等谓词生成有意义新布局，并可注入 distractor 资产库物体。
   - **Task cousins：** VLM 基于场景 affordance **提议相关新任务** 并转成仿真 goal，用于 **MimicGen 式** 演示拼接与多任务数据。

### 混合场景表示

- **前景：** 带纹理、可碰撞、可关节的 **物体 mesh**（操作交互主体）。
- **背景：** **3D Gaussian Splat**（自动管线：同视频前景擦除 + 深度监督；或手动拍摄无物体背景视频）；论文亦在部分实验使用 **Scaniverse** 等 mesh 背景。
- 项目页强调 **splat 背景 + mesh 物体** 的混合可视化与 **physics-ready** 导出。

### Real-to-sim 评测协议

- 指标：**Pearson r**（线性相关）+ **MMRV**（最坏排序违背），与 SIMPLER / PolaRiS 传统一致。
- **子任务评测：** 多步任务可按阶段拆分子成功率，论文报告可把相关从约 **0.90 提到 0.95**，便于定位瓶颈阶段指导数据采集。
- **对比：** 相对 **PolaRiS**，SimFoundry 在相同策略集上 **Pearson 平均高约 0.59**。

### Sim-to-real 训练读点

| 设置 | 代表性结果（论文/项目页） |
|------|---------------------------|
| 仅 SimFoundry 数据 | YAM Pot on Stove **99%**；DROID Stack Dishware **100%** |
| Sim + 少量 real co-train | Store Marker **60%→92%**（π₀.₅）等 |
| + object cousins | held-out 锅具任务 **+50pt** 真机增益 |
| + scene cousins | cousin 布局 Store Marker **0%→16%** |
| 多任务 + task cousins | π₀.₅-DROID 13 任务 **28%→46%**；7 held-out **0%→29%**（π₀.₅-base） |

## 源码运行时序图

官方仓 [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry)（Apache-2.0，归档见 [nvlabs-simfoundry.md](../../sources/repos/nvlabs-simfoundry.md)）按 **A 重建 → B 增广 → C 加载** 三段运行。节点对齐 README 入口：`scripts/pipeline/A_reconstruction/run.sh`、`B_augmentation/run.sh`、`C_application/run.sh`。

```mermaid
sequenceDiagram
    autonumber
    actor User as 用户
    participant A as A_reconstruction/run.sh
    participant FM as SAM3 / 深度 / Gemini VLM
    participant H3 as Hunyuan3D（s7）
    participant Pose as 位姿对齐（s8）
    participant OG as OmniGibson（s14）
    participant B as B_augmentation/run.sh
    participant C as C_application/run.sh
    User->>A: --scene-name + --video-fpath
    A->>FM: 抽帧 / 深度 / 地面 / 世界系
    FM-->>A: RGB-D + 支撑平面
    A->>FM: 迭代检物 + 分割 + inpaint
    FM->>H3: per-object crop
    H3-->>A: textured mesh
    A->>Pose: 6D 位姿
    opt --detect-articulation
        A->>A: s9 关节分解 + URDF
    end
    A->>OG: s11–14 sim-ready USD + reconstructed_og_scene.json
    User->>B: --scene-name
    B->>H3: object cousins 再生成 mesh
    B->>OG: scene variants + proposed_tasks YAML
    User->>C: --mode smoke-random / eval / demo
    C->>OG: 加载场景 / smoke / 遥操作脚手架
    Note over C: 论文级 VLA 训练与 Pearson 评测协议未随仓发布
```

- **复现先跑 A：** 桌面单平面、斜向下平移拍摄；24 GiB 卡必须 `s7_mesh.low_vram=true`。
- **C 不是论文评测：** 仓内有 smoke / teleop / demo 脚手架，但 README 写明 **数据生成与策略训练代码尚未发布**。

## 工程实践

| 项 | 内容 |
|------|------|
| 安装 | `bash scripts/installation/install_everything.sh`；Linux + NVIDIA GPU；全量约 **250 GB** |
| 服务登录 | HF 门控：SAM3 / DINOv3 / RMBG-2.0；VLM 走 Vertex AI Gemini 或 `GEMINI_API_KEY` |
| 24 GiB | `-- s7_mesh.low_vram=true`（默认 mesh 约 **29 GiB**） |
| 重建 | `scripts/pipeline/A_reconstruction/run.sh --scene-name … --video-fpath …` |
| Cousins | `scripts/pipeline/B_augmentation/run.sh --scene-name …` |
| 加载冒烟 | `scripts/pipeline/C_application/run.sh --scene-name … --mode smoke-random` |
| 示例资产 | HF [`nadunRanawaka1/simfoundry-assets`](https://huggingface.co/datasets/nadunRanawaka1/simfoundry-assets) |
| 编辑器 | `scripts/interactive/light_editor/server.py`（无需 OG）或 `run_editor.sh`（OG 内） |
| 未发布 | 论文级 data generation / policy training / evaluation |

拍摄约束（Pipeline README）：**同一平面**、斜向下、**平移不要原地旋转**、物体全程在画幅内、铰接物保持闭合。

## 评测速览

> 详见上文「核心机制 · Real-to-sim 评测协议 / Sim-to-real 训练读点」。

- **Real-to-sim 排序一致性：** 7 任务 × 5 策略族（π₀、π₀.₅、GR00T、DreamZero 等）上仿真↔真机 **均值 Pearson r=0.911、MMRV=0.018**，较 PolaRiS 平均高约 0.59。
- **Cousins 增益：** object / scene / task cousins 分别约 **+17% / +21% / +40%** 平均任务成功率；π₀.₅-base 在 7 个 held-out 任务 **0%→29%**。
- **Sim-to-real 训练：** YAM Pot on Stove **99%**、DROID Stack Dishware **100%**；π₀.₅-DROID 13 任务 **28%→46%**。

## 结论

**同一套视频孪生既要能给策略排序，也要能训出可零样本上真机的策略；digital cousins 是语义/affordance 级扩增，不是简单 pose 噪声。**

1. **Real-to-sim 排序可信** — 7 任务 × 5 策略族上仿真↔真机均值 Pearson **r=0.911**、MMRV **0.018**，较 PolaRiS 平均约高 **0.59**；子任务拆分可把相关从约 0.90 提到 0.95。
2. **Cousins 增益分层读** — object / scene / task 分别约 **+17% / +21% / +40%** 平均任务成功率；π₀.₅-base 在 7 个 held-out 任务 **0%→29%**。
3. **Sim-to-real 可操作点** — 仅 SimFoundry 数据：YAM Pot on Stove **99%**、DROID Stack Dishware **100%**；少量 real co-train 如 Store Marker **60%→92%**（π₀.₅）。
4. **管线不是零人工** — 全自动 F1 约 **0.81–0.92**；每物体约 **3 分钟** 微调可到 **0.93–0.99**，要把编辑预算算进 SLA。
5. **高 Pearson ≠ 训练免费午餐** — 排序可信后，未见物体/布局仍常需 cousins 或少量真机 demo；开源仓能重建场景，**不能**复现论文级 VLA 训练/评测数字。
6. **导出读仓不读论文口号** — 论文写 Isaac Lab；2026-08 开源默认是 **OmniGibson JSON**。

## 常见误区或局限

- **模块化 ≠ 零人工：** 全自动 F1 约 **0.81–0.92**；论文称 **每物体约 3 分钟** 微调可拉到 **0.93–0.99**——部署前应把「可接受编辑预算」算进管线 SLA。
- **与 CRISP / 人形 Real2Sim 正交：** [CRISP](../methods/crisp-real2sim.md) 面向 **人–场景接触 + 平面原语 + 人形 RL 跟踪**；SimFoundry 面向 **桌面/厨房类操作场景 + 操作臂/VLA 评测**，几何表示与下游策略接口不同，不宜混为一谈。
- **与 Agentic Real2Sim 对照：** [Agentic Real2Sim](./paper-agentic-real2sim.md)（arXiv:2607.19190）同样做真机→可仿真，但单位是 **交互 episode twin（MuJoCo 回放）**、编排层是 **VLM agent**，主指标是 **回放成功**；SimFoundry 主打 **场景孪生 + cousins + 策略 Pearson/MMRV**。
- **与 Lucida 对照：** [Lucida](./paper-lucida-r2s.md)（arXiv:2608.30821）同样产出可编辑室内物体资产，但评测停在 **检测 / ADD-SB / 场景 Chamfer**，没有 cousins 或策略相关；放置靠 **GizmoAct GUI 闭环**，不是模块化孪生栈。
- **评测相关 ≠ 训练免费午餐：** 高 Pearson 只说明 **排序可信**；策略仍可能需 **cousins 或少量真机 demo** 才能覆盖未见物体/布局（论文 co-train 与 cousins 消融已说明）。
- **代码开放度（2026-09-05）：** [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry) **部分开源**（Apache-2.0）。**已发布** A/B 管线、C 加载脚手架、light editor、HF 示例场景。**未发布** 论文级数据生成 / 策略训练 / Pearson 评测协议；自动 3DGS 背景仍标 Coming Soon。第三方权重多有门控或非商用条款。
- **论文下游 ≠ 开源下游：** 正文写 PyBullet + Isaac Lab；仓内阶段编号对齐 OmniGibson（`s14_og/`）。选型时按 README 而不是 PDF。
- **安装不是 `pip install`：** 多 conda env（`simfoundry` / `hunyuan` / 可选 `articulate`）、约 250 GB、Gemini 或 GCP。漏权或漏 `low_vram` 会在 s7 直接 OOM。

## 与其他页面的关系

- [Sim2Real](../concepts/sim2real.md) — Real2Sim 资产构建与 sim2real 训练/评测闭环
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — real-to-sim 相关性驱动的模型排序
- [Manipulation](../tasks/manipulation.md) — 操作仿真场景与 sim-ready 资产生成路线
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md) — 论文叙述的下游之一（开源默认不是它）
- [BEHAVIOR-1K / OmniGibson](./behavior-1k.md) — 开源仓实际导出与加载层
- [PhysX-Omni](./physx-omni.md) — 另一条 **统一物理字段 3D 生成** 路线，可对照「视频孪生 vs 生成式资产库」
- [Genesis World 1.0](./genesis-world-10.md) — 产业侧 **real-to-sim 评测基础设施** 叙事参照
- [GR00T N1](./paper-hrl-stack-34-gr00t_n1.md) — 实验策略族之一（GR00T N1.6/N1.7）
- [NVIDIA GEAR Lab](./nvidia-gear-lab.md) — 研究组与姊妹工作（ENPIRE、GR00T Visual Sim2Real 等）
- [Agentic Real2Sim](./paper-agentic-real2sim.md) — VLM agent 编排的 episode 级 Real2Sim（代码待开放）
- [Lucida](./paper-lucida-r2s.md) — 室内多视角 → 可编辑资产 + GizmoAct 9-DoF（几何对齐，无策略评测）
- [NVIDIA Omniverse NuRec](./nvidia-nurec.md) — 驾驶/现场神经体积 USDZ；本页是操作 mesh+cousins，不是车队日志
- [Instant NuRec](./paper-instant-nurec.md) — 前向 3DGS 驾驶世界；闭环看策略排序而非 Pearson
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页可归入其 ④ sim↔real 校准层：real-to-sim 策略评测（均值 Pearson 0.911）

## 参考来源

- [simfoundry_arxiv_2606_28276.md](../../sources/papers/simfoundry_arxiv_2606_28276.md)
- [nvidia-research-simfoundry.md](../../sources/sites/nvidia-research-simfoundry.md)
- [nvlabs-simfoundry.md](../../sources/repos/nvlabs-simfoundry.md)
- 论文：<https://arxiv.org/abs/2606.28276>
- 项目页：<https://research.nvidia.com/labs/gear/simfoundry/>
- 代码：<https://github.com/NVlabs/SimFoundry>
- 示例资产：<https://huggingface.co/datasets/nadunRanawaka1/simfoundry-assets>

## 推荐继续阅读

- PolaRiS（real-to-sim 评测对照基线，Jain et al. 2025）— 论文 Related Work 与项目页并排曲线
- SIMPLER / SimplerEnv：<https://simpler-env.github.io/> — MMRV 指标语境
- [CRISP（Contact-guided Real2Sim）](../methods/crisp-real2sim.md) — 单目视频 Real2Sim 的互补路线
- [DoorMan](./paper-doorman-opening-sim2real-door.md) — 同 GEAR 生态的 **视觉 sim2real** 姊妹工作（程序化门资产 vs 视频孪生）
- 官方仓 README / `docs/INSTALL.md`：<https://github.com/NVlabs/SimFoundry>
