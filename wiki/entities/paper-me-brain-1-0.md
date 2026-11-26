---
type: entity
tags:
  - paper
  - embodied-ai
  - memory
  - vla
  - li-auto
  - open-source
status: complete
updated: 2026-11-21
arxiv: "2609.24271"
code: https://github.com/MachEmbodied/ME-Brain-1.0
original:
  - ../../sources/papers/me_brain_1_0_arxiv_2609_24271.md
  - ../../sources/repos/me_brain_1_0.md
related:
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ./paper-me-vlm.md
  - ./paper-me-u0.md
  - ./paper-me-dex-1-0.md
  - ../overview/li-auto-machembodied-4-papers-technology-map.md
sources:
  - ../../sources/papers/me_brain_1_0_arxiv_2609_24271.md
  - ../../sources/sites/me-brain-1-0.md
  - ../../sources/repos/me_brain_1_0.md
  - ../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md
summary: "ME-Brain-1.0（arXiv:2609.24271v2，2026-09-29）：可演进记忆 + 认知核 + Focus-VLWA 动作模型；以外部记忆和技能进化闭环支持部署后持续演进；部分开源。"
---

# ME-Brain-1.0：记忆、认知与动作的自我演进具身框架

**ME-Brain-1.0**（*Memory, Cognition and Action for Evolving Embodied Intelligence*，[arXiv:2609.24271v2](https://arxiv.org/abs/2609.24271v2)，[项目页](https://machembodied.com/ME-Brain/ME-Brain-1.0.html)，[代码](https://github.com/MachEmbodied/ME-Brain-1.0)）由 **理想汽车 MachEmbodied** 提出：把 **Evolvable Memory**、**Cognitive Core** 与 **Action Model** 组成闭环——执行产生轨迹，轨迹沉淀为可检索经验，认知核在新任务中查经验、分解与重规划，动作模型在关键交互时刻生成控制，结果再写回记忆。

## 一句话定义

**机器人通过外部可演进记忆积累技能经验，在不动模型权重的前提下，让「做过的事」变成「下次能查到的知识」。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ME-Brain | MachEmbodied Brain | 本文记忆–认知–动作自我演进框架 |
| VLM | Vision-Language Model | 认知核可对接 [ME-VLM](./paper-me-vlm.md) |
| VLWA | Vision-Language-World-Action | Focus-VLWA 动作模型族 |
| SFT | Supervised Fine-Tuning | 认知/动作模块典型训练阶段 |
| SR | Success Rate | 任务成功率 |

## 为什么重要

- 纳入 [2026-09-25 理想四篇盘点](../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)：与 ME-VLM（认知训练）、ME-U0（理解–生成动作）、ME-Dex（触觉 WAM）形成同机构 **分层栈**。
- 明确 **自我演进 ≠ 在线微调权重**：演进的是记忆结构与技能库，适合长程、多任务部署叙事。
- **开源结论：部分开源**（步骤 2.5，2026-09-25）— [Focus-VLWA](https://github.com/MachEmbodied/Focus-VLWA) 动作模型训练/推理已释；完整 ME-Brain 框架、ME-VLM 子模块集成、仿真/真机栈 **待发布**。

## 核心机制

| 模块 | 职责 |
|------|------|
| **Evolvable Memory** | 编码视觉、关节、动作与结果；长轨迹事件化（抓取/放置等），归纳成功/失败模式；事件链回原始观测 |
| **Cognitive Core** | 任务分解、技能/工具调用、结果验证、失败归因与重规划；与 [ME-VLM](./paper-me-vlm.md) 训练路线对齐 |
| **Action Model（Focus-VLWA）** | 聚焦下一交互事件，结合相关历史与局部未来条件生成关节/夹爪动作；轨迹反馈记忆 |

```mermaid
flowchart LR
  mem[Evolvable Memory]
  cog[Cognitive Core]
  act[Focus-VLWA Action Model]
  env[物理环境]
  env --> mem
  mem --> cog
  cog --> act
  act --> env
  act --> mem
```

## 实验与评测

| 设定 | 结果 | 读法 |
|------|------|------|
| **Cognitive Core / embodied benchmark** | ME-VLM 35B-A3B **70.9**；领先最强对照 **8.2 分** | 26 项具身基准，14 项最佳或并列最佳 |
| **Cognitive Core / agent benchmark** | ME-VLM 35B-A3B **72.5**；领先最强对照 **9.6 分** | 4B 版本平均 **63.1**，也高于该表所有对照 |
| **Focus-VLWA / RoboMME** | **47.88%**，较 FrameSampling + Modulation 高 **3.26 个百分点** | 16 项历史依赖操作，覆盖计数、持续性、指代、模仿 |
| **Focus-VLWA / RoboDojo** | 平均 Score **21.51**、SR **16.03%**；较 π₀.₅ 高 **10.10 分 / 9.12 个百分点** | Memory 维度 Score **25.57**、SR **24.67%**；54 配置 × 24 次，共 1,296 次仿真试验 |
| **Piper 双臂真机 / ME-RealBench** | 六任务各 10 次；平均 SR **66.7%**、Score **69.5** | 相比 DM0.5 高 **11.7 个百分点 / 12.8 分**；叠碗 10/10，插充电器 1/10 |

以上均为论文 v2 报告的结果；真机 Score 为任务完成进度分，不能与成功率混为一谈。

## 与其他工作对比

| 维度 | ME-Brain 1.0 | [ME-U0](./paper-me-u0.md) | 典型端到端 VLA |
|------|--------------|---------------------------|----------------|
| 长程经验 | **显式外部记忆** + 事件检索 | 无显式长观察历史（RoboDojo 记忆维 ~7%） | 多为单窗口上下文 |
| 权重更新 | 任务级 **不改** 主模型权重 | 预训练 + 下游适配 | 微调或 RL 均可 |
| 动作生成 | Focus-VLWA 事件中心 | MoT 理解–生成联合 flow matching | 单骨干动作头 |

## 结论

**ME-Brain 1.0 把「会记住、会改计划、再动手」拆成可运维的三模块栈，用外部记忆承担演进，而不是每次任务反向传播改权重。**

1. **部分开源**：Focus-VLWA 训练与推理代码已释；ME-VLM 认知核、完整 ME-Brain 框架、仿真/真机集成和预训练权重仍待发布（按 2026-10-09 仓库 README）。
2. 真机 avg **66.7%** 说明闭环已跑通，但 **contact-rich 插入** 与 RoboDojo 低分暴露长程/精细操作上限。
3. 认知核训练见 [ME-VLM](./paper-me-vlm.md)；勿将同一系统下的 ME-VLM 案例重复计为独立真机验证集。
4. 与 ME-U0 分工：Brain 引 **过去经验**；U0 做 **当前理解 + 未来视觉–动作联合生成**。
5. 部署前区分「记忆检索延迟 / 一致性」与「动作模型频率」的工程预算。

## 源码运行时序图

图依据：基于 `MachEmbodied/ME-Brain-1.0` `README.md` 与 `MachEmbodied/Focus-VLWA@0aaf1f5` 的 `README.md` 与 `pyproject.toml`（`focus-vlwa` CLI、`HeadHistoryBuffer`、`FocusVLWAPolicy` 及 HF 权重 `zhengjingxie/focus-vlwa`）核实。

```mermaid
sequenceDiagram
  autonumber
  participant U as 开发者
  participant F as Focus-VLWA@0aaf1f5 + HF 权重
  participant Buf as HeadHistoryBuffer (20 帧)
  participant M as ME-Brain-1.0 占位仓
  participant S as 仿真 (XPolicyLab) 或真机
  U->>F: uv sync --extra train & focus-vlwa install-transformers-patch
  U->>F: focus-vlwa check-checkpoint / 下载 zhengjingxie/focus-vlwa
  opt 两阶段后训练
    U->>F: focus-vlwa post-train --stage joint -> --stage frozen
  end
  loop 控制与重规划闭环
    S->>Buf: observe(step, cam_high) 维护 25 FPS 历史缓冲
    Buf->>F: FocusVLWAPolicy.infer(3 路当前视角 + 20 帧历史)
    F-->>S: actions [50, 14] (joint flow matching)
  end
  Note over M: 完整 me_brain/ 记忆图演化 + ME-VLM 认知核仍为 TODO
```

**完整 ME-Brain 闭环：** 截至 2026-11-21，`Focus-VLWA` 动作模型代码与 HF 权重 `zhengjingxie/focus-vlwa` 已释；ME-VLM 认知核与完整 `me_brain/`/`robot/`/`simulation/` 集成仍在 Todo。

## 工程实践

| 项 | 建议与代码要点 |
|----|----------------|
| **动作模型环境与补丁** | 独立虚拟环境安装 `transformers==4.53.2`，运行 `uv run focus-vlwa install-transformers-patch` 并用 `check-checkpoint` 校验权重张量映射 |
| **头部历史缓冲契约** | `head_history` 每 25 个控制步采样 1 帧共 20 帧（每帧 16 tokens，3 路当前视角各 256 tokens，总计 1088 视觉 tokens）；每新 episode 须 `history.reset()` |
| **两阶段后训练** | 先 `--stage joint` 联合训练动作与 300M 18 层世界模型专家（`2.5e-5`，batch 288），再 `--stage frozen` 冻结 WM 专家精调动作头 |
| **仿真与真机桥接** | RoboDojo / XPolicyLab 适配器位于 `XPolicyLab/policy/FocusVLWA`；推理包使用 `asset_id="arx_x5_sim"` 对齐归一化统计 |

## 局限与风险

- **系统主仓未完整开源：** `ME-Brain-1.0` 主仓中 `me_brain/`（Evolvable Memory 外部记忆图）、`ME-VLM` 认知核及真机/仿真协同框架仍处于 TODO 状态，目前仅动作模型 `Focus-VLWA` 可独立运行。
- **长期终身记忆膨胀风险（论文 §9.2）：** 连续部署下轨迹不断入库会导致检索延迟与存储成本攀升，亟需基于任务相关性与访问频次的记忆压缩、去重与动态裁剪机制。
- **接触密集型操作瓶颈：** Piper 真机 ME-RealBench 平均 SR 66.7%，但插充电器任务仅 1/10；RoboDojo 总成功率仅 16.03%，精细插接与力控仍依赖高频局部反馈或触觉建模。
- **端侧隐私与受控消融缺口：** 论文 §9.2 指出端侧本地终身记忆隐私保护以及长程技能复用/遗忘的系统化消融尚属未来工作。

## 关联页面

- [视觉–语言–动作模型](../methods/vla.md) — 从 VLA 接口理解记忆与认知模块如何传递动作条件。
- [机器人操作](../tasks/manipulation.md) — 对照部署中遇到的新物体与技能更新任务。
- [ME-VLM](./paper-me-vlm.md)
- [MachEmbodied-U0](./paper-me-u0.md)
- [ME-Dex 1.0](./paper-me-dex-1-0.md)
- [四篇技术地图](../overview/li-auto-machembodied-4-papers-technology-map.md)

## 参考来源

- [me_brain_1_0_arxiv_2609_24271.md](../../sources/papers/me_brain_1_0_arxiv_2609_24271.md)
- [wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md](../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)
- [arXiv:2609.24271v2](https://arxiv.org/abs/2609.24271v2)
- [v2 HTML 全文](https://arxiv.org/html/2609.24271v2)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.24271)
- [Focus-VLWA 代码](https://github.com/MachEmbodied/Focus-VLWA)
- [MachEmbodied Brain 项目区](https://machembodied.com/index.html#brain)
