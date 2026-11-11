---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, loco-manip-contact-survey, humanoid, vla, teleoperation, whole-body-control, tsinghua, shanghai-pil, spirit-ai, unitree-g1]
status: complete
updated: 2026-11-11
arxiv: "2606.22174"
venue: "CoRL 2026 · Spotlight；arXiv:2606.22174v2"
code: https://github.com/OpenHLM-project/OpenHLM
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-09-vla-world-models.md
  - ../overview/loco-manip-contact-category-05-vla-world-models.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../concepts/whole-body-control.md
  - ../concepts/foundation-policy.md
  - ./gr00t-wholebodycontrol.md
  - ./paper-hrl-stack-34-gr00t_n1.md
  - ./unitree-g1.md
sources:
  - ../../sources/papers/openhlm_arxiv_2606_22174.md
  - ../../sources/sites/openhlm-project-github-io.md
  - ../../sources/repos/openhlm.md
  - ../../sources/papers/loco_manip_161_survey_154_openhlm.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
original: ../../sources/papers/openhlm_arxiv_2606_22174.md
summary: "OpenHLM（CoRL 2026 Spotlight，arXiv:2606.22174v2）：32-D 全身关节动作+双夹爪的 34-D VLA、π₀.₅ 预训练与 HuMI/stationary 异构共训；Tasks 9–11 语义泛化与 Task 12 新动作分开评估；长程水果任务 87.5% task progress（1.14h 示范），公开代码、数据和 checkpoint。"
---

# OpenHLM

**OpenHLM**（*An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation*，[项目页](https://openhlm-project.github.io/)，[代码](https://github.com/OpenHLM-project/OpenHLM)，arXiv:2606.22174）用 **单变量消融路线图** 回答：如何把语言与像素映射到人形 **全部自由度**，而不是上下身解耦的「轮式双臂」行为。产出是可复现的 **全身原生 VLA 配方**（采集接口 → 模型设计 → 异构共训），并在系统级长程任务上以 **更少演示时长** 超过 GR00T N1.6 与 Ψ₀。

> 亦收录于 [人形 Loco-Manip 161 篇](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **#154/161**（09 VLA / 世界模型）；本页以官方项目页与 arXiv 为准，策展摘要见 [survey 摘录](../../sources/papers/loco_manip_161_survey_154_openhlm.md)。

## 一句话定义

**用关节级全身遥操作 + 机器人预训练 VLA + HuMI 异构共训，做出整条运动链协调的人形 loco-manipulation recipe。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| HuMI | Humanoid-UMI（项目命名） | 无机器人、可穿戴式人手/身体示范采集 |
| WBC | Whole-Body Control | 低层全身跟踪 / 控制栈 |
| Loco-Manip | Loco-Manipulation | 行走与操作耦合的全身任务 |
| VR | Virtual Reality | PICO 等头显遥操作接口对照 |

## 为什么重要

- **把「全身原生」做成可证伪配方：** 不是又一个黑盒 checkpoint，而是三阶段 **one-variable-at-a-time** 结论——接口、预训练、共训各自贡献清晰。
- **接口决定任务可达集：** 解耦 / VR 三点 **构造上** 无法表达踩踏板、蹲身穿架；**32-D 关节全身遥操作** 是后续一切的数据前提。
- **异构共训降本：** HuMI / stationary 把 held-out 任务从 ~36% 抬到 ~84–89%，接近全全身遥操作 oracle，而无需对每个新物体再采全身遥操作；但 **HuMI 对新动作（Task 12 Pouring）并未达到同等效果**。
- **开源闭环完整：** 代码（Apache-2.0）+ HF 数据/权重 + 基于 [GR00T-WBC](./gr00t-wholebodycontrol.md) 的采集部署栈，可对照复现。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 清华大学、上海期智研究院、千寻智能（Spirit.AI） |
| 出处 | 2026-06-20 首版；2026-11-02 arXiv:2606.22174v2；CoRL 2026 Spotlight |
| 项目 / 代码 | <https://openhlm-project.github.io/> · <https://github.com/OpenHLM-project/OpenHLM> |
| 数据 / 权重 | HF `OpenHLM/OpenHLM-data` · `OpenHLM/OpenHLM-ckpts` |
| 机体 | Unitree G1（32-D 身体参考 + 2 个夹爪 = VLA 34-D；腕/头相机 + PICO4U） |
| 161 地图 | #154 · 09 人形 VLA、世界模型与通用操作 |

## 流程总览

```mermaid
flowchart TB
  subgraph phase1 [Phase 1：采集接口]
    tele["关节级全身遥操作\n32-D 身体 + 2-D 夹爪\n（对比：解耦 21-D / VR 三点 24-D）"]
    llc["低层控制器跟踪参考"]
    tele --> llc
  end
  subgraph phase2 [Phase 2：全身 VLA]
    pi["π₀.₅ 预训练初始化"]
    vla["语言 + 图像 + 本体\n→ 全身关节动作（多步 flow）"]
    pi --> vla
  end
  subgraph phase3 [Phase 3：异构共训]
    humi["HuMI 可穿戴示范"]
    stat["Stationary teleop"]
    humi --> vla
    stat --> vla
  end
  subgraph deploy [部署]
    vla --> wbc["GR00T-WBC 系低层栈"]
    wbc --> g1["Unitree G1 真机"]
  end
  llc --> vla
```

## 核心机制（归纳）

### 1）分层控制契约

高层（人 / VLA）发 **全身参考命令**，轻量低层跟踪；**接口维度**同时决定操作员表达力与 VLA 动作空间。

### 2）遥操作对照结论

| 方法 | 动作维 | 项目页结论 |
|------|--------|------------|
| Decoupled Control | 21-D | 仅部分自由度；部分任务 **N/A** |
| VR 3-Point | 24-D | 头/腕 + 导航；仍缺全身关节表达 |
| **Joint-based Whole-Body（采用）** | **32-D** | **唯一**完成对照三任务全集 |

### 3）VLA 设计消融（4 任务子集）

- **动作表示要分清：** 32-D 身体参考（root roll / pitch / yaw rate + 29 关节）再加 2 个夹爪宽度，形成 **34-D VLA 状态和动作**；π₀.₅ 的 32-D 输入/输出动作投影通过 **weight surgery** 扩为 34-D，保留旧 32 维权重。
- **例外不可忽略：** 单独更改相对动作或去本体影响较小，但论文指出**两者同时更改会造成 OOD 漂移并严重失败**，不能将单变量消融结论当成自由组合保证。

- **接口适配**（随机投影、人形排序、相对动作、去本体）单点翻转仅轻微掉点——**不是主瓶颈**。
- **预训练：** π₀.₅ ≈ **91.3%** ≫ PaliGemma **59.6%** ≫ random **41.7%** —— 跨具身机器人预训练远强于纯 VLM / 随机。
- **推理：** 多步 flow 优于 one-step / drifting（约 **+20** task-progress；尽管后者验证 MSE 可能更低）。

### 4）异构共训：三任务与四任务口径必须区分

**Tasks 9–11（新物体/语言，但复用既有运动）：** 对应官网可视化与论文 Appendix E.3/E.4。

| 设定 | Tasks 9–11 平均 Task Progress |
|------|-------------------------------|
| 仅 Tasks 1–8 全身遥操作 | 36% |
| + HuMI | **84%** |
| + Stationary teleop | **89%** |
| 11 任务全身遥操作 oracle | 96% |

**Tasks 9–12（四个全部 held-out，包含新动作 Pouring）：** 仅 8-task baseline **33%** → 加 4 项 stationary teleop **87%**，12-task 全身遥操作 oracle **94%**（论文 §1、§3.3）；HuMI 对 **Task 12 新的倾倒/旋转动作表现失败**，不能把上表 84% 的三任务成绩称为「四任务平均」。

- **实际采集成本（每个 held-out 任务 40 条）：** 全身遥操作约 21min、stationary 约 13min、HuMI 约 7min；其中 full-teleop 额外重置/过热成本尚未算入。
- **迁移边界：** HuMI 的相机视角/夹爪差和 IK 关节动作分布差，使其现阶段更擅长迁移语义而非全新动作。

## 实验与评测

| 轴 | 口径（项目页） |
|----|----------------|
| 长程果盘（20 种有序果对，**选 10 组实机评估**） | OpenHLM **87.5%** @ **1.14 h** vs GR00T N1.6 **57.5%** / Ψ₀ **48.8%** @ **2.70 h**（均为 task progress，非 success rate） |
| 12 任务四类 | Pick&Place / Workspace / Body-as-Tool / Constraints；官网称平均 progress **>90%**；论文内部 8 个训练任务平均 **89%**，分组与协议不同 |
| 户外 | 约 **20** 条户外 demo 混入实验室数据 → 校园清扫类自主行为（官方演示，非定量对照 benchmark） |

## 开源状态（项目页及代码仓，2026-11-11 核查）

| 产物 | 状态 |
|------|------|
| 代码 | [OpenHLM-project/OpenHLM](https://github.com/OpenHLM-project/OpenHLM) · Apache-2.0 |
| 数据 | HF `OpenHLM/OpenHLM-data` |
| 权重 | HF `OpenHLM/OpenHLM-ckpts` |
| 低层栈 | 仓内 GR00T-WBC 改写块 + openpi 训练/服务 |

## 源码运行时序图

节点对齐 [`sources/repos/openhlm.md`](../../sources/repos/openhlm.md)。

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant WBC as GR00T-WholeBodyControl<br/>4-OpenHLM
    participant HuMI as HuMI4OpenHLM
    participant OP as openpi4OpenHLM
    participant HF as HF OpenHLM/<br/>data · ckpts
    participant G1 as Unitree G1
    alt 数据采集
        U->>WBC: VR 全身遥操作 / 同步采集脚本
        WBC->>G1: 低层跟踪 + 传感器录制
        U->>HuMI: Vive tracker + 腕部 GoPro 采集
        HuMI-->>OP: final_data 共训片段
    else 训练
        U->>OP: convert_*_to_lerobot + compute_norm_stats
        OP->>HF: 或本地 LeRobot 数据集
        U->>OP: torchrun train_pytorch.py config
        OP-->>U: 训练 checkpoint
    else 部署
        U->>OP: serve_policy.py --env SONICG1
        U->>WBC: deploy_stream.sh + openpi-eval/main.py
        OP-->>WBC: websocket 动作
        WBC->>G1: 全身关节命令
    end
```

关键复现路径：先通 GR00T-WBC 部署/遥操作文档 → 用示例示范跑 LeRobot 转换与 `openhlm_example` 训练 → `serve_policy` + `deploy_stream` 真机闭环。已核对官方 README 与仓内 `train_pytorch.py` / `serve_policy.py` / `deploy_stream.sh` 入口；**本次未实际运行训练或真机闭环**。

## 工程实践

| 项 | 要点 |
|----|------|
| 硬件 | G1 + ChangingTek 腕夹爪 + D405 腕摄 + SV1-25 头摄 + PICO4U |
| 训练入口 | `src/openpi4OpenHLM/scripts/train_pytorch.py` |
| 部署入口 | `serve_policy.py` + `GR00T-…/scripts/deploy_stream.sh` |
| 检查点示例 | `12tasks_*` / `20fruit-arrangement_*`（含 full-teleop / HuMI / stationary 变体） |
| 论文 v2 训练口径 | AdamW，30k steps，global batch 128，50-step action horizon，224×224；4×A800 约 24h（作者报告） |
| 推理节奏 | 30Hz 控制下输出 50-step chunk，最多执行 25 步后重推；flow sampler 10 steps，不等于策略每 30Hz 重推 |
| 真机安全 | 官方 README 提醒先仿真遥操作再上机、低增益启动；本文未执行硬件实验 |

## 结论

**全身原生 loco-manipulation 的可证伪配方是：32-D 关节遥操作接口 → 机器人预训练 VLA → HuMI/stationary 异构共训。**

1. **接口是前提** — 解耦 21-D / VR 三点 24-D 构造上无法覆盖踩踏板、蹲身穿架；只有 32-D 关节全身完成对照三任务全集。
2. **预训练远强于适配技巧** — π₀.₅ 约 91.3% ≫ PaliGemma 59.6% ≫ random 41.7%；随机投影等人形适配单点翻转不是主瓶颈。
3. **推理选多步 flow** — 十步 91.3% 对照单步 flow 72.9% / drifting 70.8%，虽单步延迟由约 90ms 降至 60ms（作者 RTX 5080），但验证 MSE 低不等于真机更稳。
4. **异构共训有边界** — Tasks 9–11 的 36% → HuMI 84% / stationary 89% / oracle 96% 仅适用于运动复用；包含 Pouring 的 Tasks 9–12，baseline 33% → stationary 87% / oracle 94%，HuMI 未学会新增倾倒动作。
5. **系统级读数** — 20 种水果有序组合中抽测 10 组，OpenHLM 87.5% @ 1.14h 示范，对照 GR00T N1.6 57.5% 与 Ψ₀ 48.8% @ 2.70h；均为 task progress，不能替换为成功率。
6. **工程代价** — 开源码/数据/权重，但真机依赖 GR00T-WBC / TensorRT / VR 标定；户外结果偏定性，勿与 161 策展旧摘要等同。

## 局限与风险

- **配方依赖全身遥操作质量**；解耦接口无法「事后补自由度」。
- **真机栈重**：依赖 GR00T-WBC / TensorRT / VR 标定，工程门槛高于「只下权重」。
- **户外结果偏定性**；系统级数字以室内长程果盘与 12 任务为主。
- **真实世界样本与泛化边界：** HLM-12 每个任务/策略仅 5 次 rollout；论文只在 G1 验证，跨机体泛化未经证明。长程系统比较虽有相同演示预算对照，但 GR00T 和 Ψ₀ 为适配同一硬件/控制接口的版本，不能外推为原生模型通用排名。
- **必须区分指标：** 论文表中是按 rubric 计算的 task progress（含中间步骤得分），不等于二值任务完成率；验证 action MSE 不能替代真机评估。
- **勿与 161 策展旧摘要等同：** 公众号条目为地图坐标；消融与开源以本页与官方仓为准。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| **GR00T N1.6 / Ψ₀** | 项目页系统级基线；OpenHLM 以 **更少演示时长** 取得更高长程 progress |
| **[GR00T-WholeBodyControl](./gr00t-wholebodycontrol.md)** | 低层采集/部署栈上游；OpenHLM 仓内为其改写块，非替代 SONIC 训练 |
| **解耦上下身 VLA** | 仅暴露部分自由度；OpenHLM 强调 **32-D 关节全身** 才可达踩踏板/蹲穿架等任务 |
| **[HumanoidArena](./paper-humanoidarena.md)** | 仿真分层 **GMT 接口基准**；OpenHLM 是真机 **全身原生 VLA 配方**——可对照中间动作 vs 全关节动作空间 |
| **HuMI / UMI 系** | OpenHLM 把 HuMI 用作 **异构共训** 降本通道，而非唯一数据源 |

## 关联页面

- [Loco-Manip 接触分类 05：VLA 与世界模型调用](../overview/loco-manip-contact-category-05-vla-world-models.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 全身移动操作任务坐标
- [VLA](../methods/vla.md) — π₀.₅ / GR00T 等对照语境
- [Teleoperation](../tasks/teleoperation.md) — 全身遥操作与 HuMI 采集
- [GR00T-WholeBodyControl](./gr00t-wholebodycontrol.md) — 低层控制与部署上游
- [Foundation Policy](../concepts/foundation-policy.md) — 人形基础策略分层叙事
- [161 分类 hub](../overview/loco-manip-161-category-09-vla-world-models.md) — 策展地图位置

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- [OpenHLM v2 正文 + Appendix A–E 通读记录（arXiv:2606.22174v2）](../../sources/papers/openhlm_arxiv_2606_22174.md)
- [OpenHLM 项目页归档](../../sources/sites/openhlm-project-github-io.md)
- [OpenHLM 代码归档](../../sources/repos/openhlm.md)
- [161 策展摘录 #154](../../sources/papers/loco_manip_161_survey_154_openhlm.md)

## 推荐继续阅读

- [OpenHLM 项目页](https://openhlm-project.github.io/) — 视频、12 任务与消融数字
- [arXiv:2606.22174v2 全文 HTML](https://arxiv.org/html/2606.22174) — 正文 §1–6 与附录 A–E
- [GR00T-WholeBodyControl 文档](https://nvlabs.github.io/GR00T-WholeBodyControl/) — 低层栈安装对照
