---
type: entity
project_id: seebfm
project: https://capfab.github.io/seebfm-media/
tags: [paper, humanoid, behavioral-foundation-model, forward-backward, unsupervised-rl, terrain-adaptation, depth-perception, sim-to-real]
status: draft
updated: 2026-11-11
venue: "2026 · preprint / under review (作者主页)"
original:
  - ../../sources/papers/seebfm_preprint_2026.md
sources:
  - ../../sources/papers/seebfm_preprint_2026.md
  - ../../sources/sites/seebfm-project-page.md
  - ../../sources/repos/seebfm-media-page-source.md
related:
  - ./paper-bfm-zero.md
  - ./paper-p-bfm.md
  - ./paper-perceptive-bfm.md
  - ../concepts/behavior-foundation-model.md
  - ../../roadmap/depth-bfm.md
summary: "SeeBFM 在 BFM-Zero 上以地形高度扫描后训练：行为 prompt 的 backward 表征保持地形无关，控制策略依据地形变化执行；双深度相机学生网络预测高度扫描并替代特权扫描输入。官方演示只有仿真结果，完整论文尚未取得，算法代码与真机验证待核查。"
---

# SeeBFM：能“看”的行为基础模型 / Behavior Foundation Model That Sees

**项目身份：** SeeBFM（2026，作者主页标记 *under review*）；作者 Thanh Ly、Truong-Duy Dang、Hung Nguyen、Duy M. H. Nguyen、Vien Anh Ngo、An T. Le。研究单位（按官方演示片头）：VinRobotics、UC San Diego、DFKI、University of Stuttgart、IMPRS-IS、CAIR（VinUniversity）、IAS（TU Darmstadt）。

**一手资料：** [项目主页](https://capfab.github.io/seebfm-media/) · [官方仿真回放](https://capfab.github.io/seebfm-media/playground/) · [作者论文列表](https://anindex.github.io/) · [网站与回放源码](https://github.com/capfab/seebfm-media)。

> **通读状态：待补全文。** 目前已通读官方讲解分镜 `slides/spec.js`（15 段）及 `overview.vtt`、核对公开回放源码；尚未取得完整论文 PDF/附录，故本页为草稿。下文数字是官方演示的报告值，不等同于独立复核过的论文实验表格。

## 一句话定义

**SeeBFM 用高度扫描让预训练 BFM-Zero 在楼梯、箱体等地形下调整动作，同时保持奖励、目标姿态与运动参考所表示的行为意图；再用双深度相机学生模型预测扫描，替换仿真中直接提供的地形输入。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| SeeBFM | Behavior Foundation Model That Sees | 以地形扫描进行后训练、以深度相机感知输入部署的行为模型 |
| BFM | Behavioral Foundation Model | 复用多种提示接口与全身行为的基础模型 |
| FB | Forward–Backward Representation | 将未来状态与行为意图映射进共享 latent；本文保持 backward 网络不读取地形 |
| RL | Reinforcement Learning | 地形后训练延续 BFM-Zero 的强化学习框架 |
| G1 | Unitree G1 | 作者计划用于真实部署测试的人形平台 |

## 为什么重要

在平地学到的行为 prompt（例如“以 0.7 m/s 向前走”或“达到目标姿态”）不包含台阶高度、落脚点、避障净空等信息。若控制器只能读自身状态，同一个意图到了楼梯就可能摔倒。SeeBFM 的设计问题不是再定义一次“向前走”，而是**让当前地形改变动作实现方式，而不改变 prompt 原本要实现的目标**。

## 核心原理

### 1. BFM-Zero 的可提示行为意图

原 BFM-Zero 的 forward–backward（FB）表征支持由 reward、goal pose、motion 获得行为 latent，并由一个条件化全身控制策略执行。SeeBFM 从这个预训练控制器继续学习，而不是将它替换为单一上楼梯策略。

### 2. 地形扫描后训练与条件化边界

官方视频解释：在复杂地形继续训练 BFM-Zero 时，各相关网络输入地面高度扫描，**唯独将 reward / goal pose / motion 转换为 prompt 的 backward 网络不读地形扫描**。因此相同奖励/目标映射出的行为意图不因局部地形改变；策略可通过扫描调整步态、摆动腿抬高与躯干动作。

这一网络位置的选择是论文主张之一，不能误读为“所有网络都拼接相机图像”或“通过语言理解直接决定落脚点”。具体 FB 目标函数、损失权重、训练环境与扫描形状，仍须完整论文确认。

### 3. 深度学生替换特权高度扫描

离线录制 SeeBFM 在仿真地形上的控制轨迹，渲染两个深度摄像头观测并训练学生预测相应地形高度扫描。推理时由学生预测值替代真实高度扫描，**已有控制策略保持冻结**。这是“感知估计器 → 高度扫描接口 → 控制策略”的模块化替换，并不是有了深度相机就自动完成 Sim2Real。

### 4. Typed prompt 的边界

官方演示介绍将打字指令映射到一个运动 prompt；其公开 Playground 并非开放词汇真机控制：浏览器使用 MiniLM 嵌入在已录制的句子中搜索近邻，展示对应预录动作。因此网页可交互**不代表**在线运行任意语言指令下的 SeeBFM 策略。

## 流程总览

以下模块关系来自已完整读取的官方解说；图中未填入无法从公开材料验证的损失函数、网络层数及训练超参数。

```mermaid
flowchart TB
    A["BFM-Zero 预训练行为基础模型"] --> B["复杂地形 + 高度扫描后训练"]
    P["奖励 / 目标姿态 / 参考运动"] --> Z["Backward 网络生成行为 latent z"]
    B --> C["接收地形扫描的控制网络与其他学习模块"]
    Z --> C
    H["仿真地形真实高度扫描"] --> C
    C --> R["全身动作 / 仿真 rollout"]
    R --> D["录制轨迹及双深度相机观测"]
    H --> E["深度学生的高度扫描监督目标"]
    D --> S["双相机深度学生模型"]
    E --> S
    S --> HP["学生预测的高度扫描"]
    HP --> F["冻结的 SeeBFM 控制策略"]
    Z --> F
    F --> ACT["地形条件化的全身动作"]
```

**关键设计：** 高度扫描进入控制路径，但不进入 backward prompt 表征；感知学生训练独立于已冻结的控制策略。流程表示项目公开材料可核实的阶段，不承诺开放源码中存在可运行的对应入口。

## 工程实践

| 环节 | 已公开信息 | 复现现状 |
|---|---|---|
| 行为接口 | reward / goal pose / motion；视频另有 typed 指令演示 | 可在网页查看提示效果；算法接口代码待公开 |
| 地形输入 | 机器人周围的高度扫描 | 未核实扫描维度、采样频率、噪声模型 |
| 视觉输入 | 两个深度摄像头，学生回归高度扫描 | 没有查到可下载学生权重/训练入口 |
| 控制器 | BFM-Zero 后训练得到的 SeeBFM，摄像头替换时冻结 | 没有找到 SeeBFM 策略实现与权重 |
| 评测 | 官网 Isaac Lab 预录仿真回放，包含跌倒、停滞和穿越 | Playground 可重复看录制结果，不是实时仿真 |
| 真机 | 作者计划在 G1 上用深度学生做测试 | 2026-11-11 尚无完成验证的官方证据 |

### 公开演示代码的定位

[capfab/seebfm-media](https://github.com/capfab/seebfm-media) 目前含静态项目网页、`media/`、`slides/`、`playground/`、`playground/data/`，其中 `viewer.js`/ `terrain.js`/ `fk.js` 只用于将**预录的仿真**机器人轨迹显示成 3D 场景；它不是策略 actor、critic、相机学生的训练或推理代码。

## 源码运行时序图

**控制策略训练/推理的源码时序图：不适用（相关官方运行入口未公开）。** 下图仅为已读取的**项目网站 Playground 回放源码**的真实调用路径，不能当作 SeeBFM 策略推理流程。

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant H as playground/index.html
    participant V as playground/viewer.js
    participant J as data/index.json + robot.json/bin
    participant T as playground/terrain.js
    participant A as data/a/*.bin.gz
    participant K as playground/fk.js
    participant R as Three.js Renderer
    U->>H: 打开 Playground
    H->>V: import viewer.js
    V->>J: fetch 场景索引和机器人模型
    J-->>V: models / prompts / cells / attempt 映射
    U->>V: 选模型、prompt、地形、录制尝试
    V->>T: loadTile() 读取 cells 地形网格
    V->>A: loadAttempts() 读取预录动作
    A-->>V: 根位姿、关节与事件时间序列
    loop 浏览器渲染每帧
        V->>V: frame() 轨迹插值
        V->>K: forward() 前向运动学
        K-->>V: 刚体位姿
        V->>R: 更新机器人/地形 Mesh，render()
    end
```

重现网页交互仅需访问 `playground/index.html` 并使用公开素材，**不能**复现高度扫描后训练、相机学生训练或真实机器人执行。

## 实验与评测

**实验范围：** 以下是官方视频 `slides/spec.js` 与字幕 `media/overview.vtt` 所报告的仿真结果，口径为“穿越地形（crossing）”，与“未摔倒（upright）”或“按要求完成目标动作”不是同一个指标。对照组包含原 BFM-Zero 与高度扫描消融。

| 地形（六个 locomotion prompts） | SeeBFM 穿越率 | BFM-Zero 穿越率 |
|---|---:|---:|
| 上楼梯 | 54.4% | 0% |
| 箱体 | 41.0% | 0% |
| 下楼梯 | 68% | 5% |

- **消融：** 上楼梯 SeeBFM 为 54.4%；将地形扫描固定为水平面时 4.0%；训练时省略扫描为 0%（官方演示值）。
- **速度保持：** 官方示例跨地形的 forward / turn / sideways / backward prompt 对应 76% / 72% / 42% / 12% 的速度保留比例；向后行走明显趋于停滞，不能用“没有跌倒”替代任务完成率。
- **目标姿态：** 在地形上保持相同目标姿态 prompt，终点相对平地额外偏离约 2–4 cm（官方解说口径）。
- **障碍边界：** 公开示例中面对 45 cm 方块停在前方；演示明确称高于约 35 cm 时机器人不尝试穿越。
- **语言行为：** 官方解说称上/下楼梯及箱体的常见 typed request 约一半尝试可完成；未从论文获取更细粒度表格或测试分布。

以上数字未经完整论文正文、附录和独立实验复核；**不能据此报告真实世界成功率**。

## 结论

**SeeBFM 的可迁移设计是固定行为意图的解释层，让地形仅改变执行策略，并通过扫描预测学生将特权地形输入替换成双相机输入；当前验证仍主要是仿真展示。**

1. **表征边界先行：** backward 网络不接地形，减少同一 reward 或动作指令在不同地形语义漂移。
2. **模块化视觉替代：** 先训地形 scan 控制器，再以双深度学生估计 scan；可以冻结控制器，而非联合重训全部网络。
3. **“越过”和“站稳”分开评价：** 同时计 crossing、stayed short、fell，防止策略靠停下“刷出”稳定结果。
4. **prompt 类型不能视作等价：** 前进/转向迁移较好；侧走/后退速度保持下降，应逐类评估。
5. **部署结论留白：** 现有公开演示尚无 G1 搭载深度学生的真实部署结果，也未提供算法训练入口；待论文全文与实现开放后复核。

## 与其他工作对比

| 工作 | 地形适配入口 | 保持行为语义的方式 | 公开验证状态 |
|---|---|---|---|
| [BFM-Zero](./paper-bfm-zero.md) | 主要是平地、依赖本体状态 | FB latent 统一 reward / goal / motion | 原工作含真机；直接上复杂地形受限 |
| **SeeBFM** | 后训练接入高度扫描；双相机学生回归 scan | backward 不读地形，其他网络可据地形执行 | 已公开仿真演示，G1 视觉部署待测 |
| [P-BFM](./paper-p-bfm.md) | 深度条件化无监督 RL + 显式根指令 ERI | 区分地形无关意图与条件化执行 | 原文给出 G1 仿真与真机 |
| [Perceptive BFM / PMT](./paper-perceptive-bfm.md) | 通过地形一致运动参考和感知校正 | 保留原始 motion reference 接口 | 原论文为不同的训练与跟踪路线 |

## 局限与风险

- **全文缺失：** 作者主页目前只有 under review 状态；论文 PDF、数学推导、完整实验表与附录尚未取得，本页 status 保持 `draft`，健康度原文项不计分。
- **算法未开源：** `capfab/seebfm-media` 为网站/数据可视化源码，不可当作 SeeBFM 的训练代码、checkpoints、深度估计学生或可上机 runtime。
- **仅仿真证据：** G1 搭载双深度学生的真机验证处于筹备阶段；缺少真实相机延迟、遮挡、动态地形、Sim2Real 成功率测量。
- **停滞模式：** 对较高障碍和向后移动出现“停在前方而不跌倒”；任务完成应与稳定性分别报告。
- **数字可比性：** 网站演示给出汇总百分比，尚未从完整论文核实独立 seed、试验数、分母、成功判定阈值及误差区间，不据此推断统计显著性。

## 关联页面

- [BFM-Zero](./paper-bfm-zero.md) — SeeBFM 继承的 promptable FB 基座
- [P-BFM](./paper-p-bfm.md) — 同样关注感知地形适配，但引入显式根指令
- [Perceptive BFM / PMT](./paper-perceptive-bfm.md) — 感知运动跟踪路线对照
- [Behavior Foundation Model 概念](../concepts/behavior-foundation-model.md) — 总体分类与主干知识
- [BFM 纵深路线](../../roadmap/depth-bfm.md) — 学习路径

## 参考来源

- [SeeBFM 论文来源与待补核查](../../sources/papers/seebfm_preprint_2026.md)
- [SeeBFM 官方项目及公开视频](../../sources/sites/seebfm-project-page.md)
- [SeeBFM 网页回放源码核查](../../sources/repos/seebfm-media-page-source.md)
- [作者主页](https://anindex.github.io/) — 作者、状态与项目入口
- [官方视频分镜/字幕](https://github.com/capfab/seebfm-media/blob/main/slides/spec.js) — 方法讲解、实验示例与口播数值

## 推荐继续阅读

- [SeeBFM 官方项目与 Playground](https://capfab.github.io/seebfm-media/) — 地形和 prompt 的公开仿真演示
- [BFM-Zero 原始论文（arXiv:2511.04131）](https://arxiv.org/abs/2511.04131) — FB prompt 与预训练基座
- [P-BFM 原始论文（arXiv:2610.15833）](https://arxiv.org/abs/2610.15833) — 独立的感知行为基础模型路线
