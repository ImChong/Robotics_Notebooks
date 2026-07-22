---
type: entity
tags:
  - paper
  - vln
  - outdoor-navigation
  - city-scale
  - vision-language-action
  - vision-language-model
  - chain-of-thought
  - recovery
  - sim2real
  - quadruped
  - humanoid
  - carla
  - shanghaitech
  - ict
  - casia
  - ucas
  - leju
status: complete
updated: 2026-07-22
arxiv: "2607.11638"
summary: "DA-Nav（arXiv:2607.11638）：用商业导航方向指令做城市尺度户外 VLN；图像平面离散 spatial grounding + CoT 偏离恢复；ReDA 数据集；CARLA SoTA（CSR≈98%），零样本 Go2 / 乐聚 Kuavo-V 公里级导航；截至入库日未开源。"
related:
  - ../tasks/vision-language-navigation.md
  - ../concepts/sim2real.md
  - ../methods/vla.md
  - ../overview/vln-open-source-repro-paradigms.md
  - ./paper-notebook-navila-legged-robot-vision-language-action-model.md
  - ./paper-worldvln-aerial-vln-wam.md
  - ./paper-realm-last-3-meter-vln-grounding.md
  - ./paper-uni-lavira.md
  - ../methods/behavior-cloning.md
sources:
  - ../../sources/papers/da_nav_arxiv_2607_11638.md
---

# DA-Nav（方向感知城市尺度 VLN）

**DA-Nav**（*Direction-Aware City-Scale Vision-Language Navigation*，[arXiv:2607.11638](https://arxiv.org/abs/2607.11638)，上海科技大学 / 中科院计算所 / 中科院自动化所 NLPR / 国科大 / XYZ Embodied AI）把城市尺度户外导航写成 **Direction-Aware VLN**：用 **Google Maps / 高德** 等商业工具给出的粗粒度方向指令，配合 egocentric RGB，经 **VLM 在图像平面离散网格上 grounding**，并以 **Chain-of-Thought（偏离评估→动作→网格轨迹）** 支撑长程闭环恢复。配套 **ReDA**（方向指令 + recovery 轨迹）。仿真相对 CityWalker / ViNT / NaVid / NaVILA 等报告更强成功率与纠偏；真机 **零样本** 部署 Unitree Go2 与乐聚 **Kuavo-V**，展示公里级户外闭环。

## 一句话定义

**把商业导航的稀疏方向提示，落地为「图像平面离散目标 + CoT 纠偏」的户外 VLN 策略**——避开稠密地图与细粒度语言监督，同时显式训练偏离恢复。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 视觉–语言条件下的具身导航 |
| CoT | Chain-of-Thought | 结构化逐步推理（本文：偏离→动作→网格） |
| VLM | Vision-Language Model | 视觉–语言骨干（本文 Qwen2.5-VL-7B） |
| ReDA | Recovery + Direction-Aware（dataset） | 方向指令与恢复轨迹数据集 |
| CSR | Correction Success Rate | 偏离后成功纠回比例 |
| DF | Deviation Frequency | 单位行程偏离事件频率 |
| IPM | Inverse Perspective Mapping | 无可靠深度时的平面假设回退投影 |
| LoRA | Low-Rank Adaptation | 冻结骨干上的参数高效微调 |

## 为什么重要

- **城市尺度可扩展监督：** 相对 R2R 式细粒度描述或地标图，商业导航指令已规模化存在；难点在 **人读方向 → 机器人局部动作**。
- **动作表示对齐 VLM：** 连续 3D waypoint 回归与自回归 VLM 能力错位；离散 **图像平面网格** 把决策落在视觉推理更稳的空间。
- **闭环恢复是一等公民：** 纯专家 BC 在 OOD 偏离下 CSR 崩溃（消融 15.46%）；ReDA + CoT 把 CSR 拉到约 **98%**。
- **跨具身零样本信号：** 同一策略从 CARLA 迁到 **四足 / 人形**，无需真机微调，对户外 [Sim2Real](../concepts/sim2real.md) 与足式导航栈选型有参考价值。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 上海科技大学（ShanghaiTech）；中国科学院计算技术研究所（ICT）；中国科学院自动化研究所（CASIA / NLPR）；中国科学院大学（UCAS）；XYZ Embodied AI |
| arXiv | [2607.11638](https://arxiv.org/abs/2607.11638)（v2，2026-07-14） |
| 项目页 / 代码 | **未开源**（截至 2026-07-22；abs/HTML/PDF 无 GitHub / 项目页） |
| 骨干 | Qwen2.5-VL-7B-Instruct + LoRA（视觉编码器冻结） |
| 仿真 | CARLA 0.9.15；训练/域内 Towns 01–05、10HD；零样本 Towns 06/07/15 |
| 真机 | Unitree Go2；乐聚 Kuavo-V；Intel RealSense D455；远端 RTX 4090 推理 |
| 主要基线 | CityWalker、ViNT、NaVid、NaVILA、零样本 Qwen2.5-VL |

## 核心原理

### 输入 / 输出

| 侧 | 内容 |
|------|------|
| 视觉 | egocentric RGB 历史 \(\mathbf{O}_t=(o_{t-k},\dots,o_t)\)，\(k=4\) |
| 语言/指令 | 离散方向 \(I_t\in\{\texttt{FORWARD},\texttt{TURN\_LEFT},\texttt{TURN\_RIGHT},\texttt{STOP}\}\) |
| CoT 输出 | \(Y_t=(s_t,c_t,\mathbf{P}_t)\)：是否偏离、高层动作、\(L=6\) 网格点序列 |
| 执行 | 网格→机体系连续轨迹 \(\mathbf{W}_t\) → furthest-point 角速度/线速度 |

### 流程总览

```mermaid
flowchart LR
  Cam[Egocentric RGB 历史] --> VLM[Qwen2.5-VL-7B + LoRA]
  Nav[商业导航方向指令] --> Parse[指令解析模块]
  Parse --> VLM
  VLM --> CoT["CoT: s_t → c_t → P_t 网格"]
  CoT --> Proj[深度或 IPM 投影到机体系]
  Proj --> FPC[Furthest-point 控制]
  FPC --> Robot[四足 / 人形低层控制]
```

### 关键机制（压缩）

1. **离散网格：** 可通行区 \(G=\{(r,c)\mid r\in[13,23],\,c\in[0,28]\}\)；未来约 3 s 轨迹写成 6 个网格目标。
2. **结构化生成：** prompt 注入网格约束，强制先判断偏离再选动作，再预测 \(\mathbf{P}_t\)，减轻空间幻觉与决策不一致。
3. **ReDA 恢复数据：** CARLA 三态 FSM（Stable / Drifting / Recovering）；\(e_y\geq 0.35\) m 进入恢复；丢弃 Drifting；约 286k 帧（含 128k recovery）。
4. **控制接口：** 取视界最远预测点作目标；转向饱和 + 大转向时降速，缓解 VLM 自回归延迟引起的高频抖动。

## 源码运行时序图

**不适用**：截至 **2026-07-22**，arXiv 页面与公开检索均未确认官方可运行仓库、权重或项目页；无法对齐 README 训练/部署入口绘制复现时序。若后续开源，应补 `sources/repos/` 与本图。

## 工程实践

| 项 | 建议 / 论文设定 |
|----|----------------|
| 指令接口 | 手机侧异步解析商业导航文案 → 离散 \(I_t\)；勿假设可直接喂长自然语言 |
| 感知 | 前视 RGB 即可；深度可用则优先投影，否则 IPM（平地假设） |
| 部署算力 | 文中真机将 **7B VLM** 放远端 GPU（例 RTX 4090），机载流式观测/接收决策 |
| 评测勿只看 SR | 并列报告 **DF / CSR**；否则会低估「偶发偏离后能否回来」 |
| 复现现状 | **无官方代码/权重**；仅能作方法选型与对照，不能当可跑栈 |

## 实验要点（索引级）

| 设置 | 结果要点 |
|------|----------|
| CARLA 239 轨迹（Table II） | DA-Nav SR **59.00%**、SPL **58.66**、CSR **98.15%**；NaVid/NaVILA CSR≈**5%** |
| 摘要未见环境 | 未见城镇 SR **56.16%**（与表内总体 59% 并存，以 PDF 分城镇图为准） |
| 消融（Table III） | w/o recovery：SR 29.71%、CSR 15.46%；w/o CoT：SR 38.91%、DF 4.30 |
| 真机开环原语 | 平均 SR **83.3%**（直/左/右各 12） |
| 真机闭环 | 城市+公园整体 SR **46.7%**（CityWalker 23.3%、ViNT 16.7%） |
| 跨具身 | Kuavo-V **零样本** 连续 **1.2 km** 户外目标导航 |

## 局限与风险

- **依赖商业导航 / GPS：** 作者自述受 GPS 多径、更新延迟与未覆盖区域限制；未来拟走向自主拓扑记忆。
- **未开源：** 无 ReDA 与权重公开入口，工程复现门槛高。
- **算力与延迟：** 7B 自回归推理需远端 GPU；控制侧用 furthest-point 缓解，但实时性仍是部署约束。
- **平地 IPM 回退：** 崎岖/坡道上几何误差会传导到机体系轨迹。
- **误区：** 把 DA-Nav 等同于「又一个室内 R2R VLN」——其指令接口与评测主线是 **户外方向感知 + 恢复**，与 Matterport 离散图栈不同。

## 与其他工作对比

| 路线 | 指令形态 | 动作表示 | 恢复监督 | 开源/复现 |
|------|----------|----------|----------|-----------|
| **室内 VLN 四范式**（VLFM→Uni-NaVid） | 细粒度语言等 | 地图 / LLM / 扩散 / 离散命令 | 多为专家轨迹 | [开源复现路径](../overview/vln-open-source-repro-paradigms.md) |
| **NaVILA** | 混合语言 | 分层 VLA→足式低层 | 弱纠偏（文中 CSR≈5%） | 见 [笔记实体](./paper-notebook-navila-legged-robot-vision-language-action-model.md) |
| **CityWalker / ViNT** | 路径/拓扑目标 | 连续 waypoint / 拓扑 | 弱显式恢复 | 基线对照 |
| **WorldVLN** | 空中语言指令 | WAM 潜转移→航点 | 侧重世界预测 | [空中 WAM](./paper-worldvln-aerial-vln-wam.md) |
| **DA-Nav（本文）** | **商业方向离散指令** | **图像平面网格 + CoT** | **ReDA recovery** | **暂未开源** |

## 关联页面

- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md) — 任务总览；本页补 **城市尺度方向感知** 分支
- [Sim2Real](../concepts/sim2real.md) — CARLA→足式/人形零样本户外导航
- [VLA](../methods/vla.md) — VLM/VLA 动作表示与导航子任务
- [VLN 四范式开源复现](../overview/vln-open-source-repro-paradigms.md) — 可跑通栈对照（本文暂不可跑）
- [NaVILA](./paper-notebook-navila-legged-robot-vision-language-action-model.md) — 足式导航 VLA 基线
- [WorldVLN](./paper-worldvln-aerial-vln-wam.md) — 另一城市/户外相关导航范式（空中 WAM）
- [REALM](./paper-realm-last-3-meter-vln-grounding.md) — 室内 REVERIE 末段接地（评测关注点不同）
- [行为克隆](../methods/behavior-cloning.md) — 纯专家示范在闭环 OOD 下的局限对照

## 参考来源

- [DA-Nav 论文摘录（arXiv:2607.11638）](../../sources/papers/da_nav_arxiv_2607_11638.md)

## 推荐继续阅读

- Yuan et al., *DA-Nav: Direction-Aware City-Scale Vision-Language Navigation* — [arXiv:2607.11638](https://arxiv.org/abs/2607.11638)
- Cheng et al., *NaVILA: Legged Robot Vision-Language-Action Model for Navigation* — RSS 2025（文中主要 VLA 基线）
- Liu et al., *CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos* — CVPR 2025（连续 waypoint 对照）
- Shah et al., *ViNT: A Foundation Model for Visual Navigation* — CoRL 2023
- Bai et al., *Qwen2.5-VL Technical Report* — [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)（本文骨干）
