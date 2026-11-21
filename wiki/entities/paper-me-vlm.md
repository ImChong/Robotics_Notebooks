---
type: entity
tags:
  - paper
  - vlm
  - embodied-ai
  - agent
  - li-auto
status: complete
updated: 2026-11-21
arxiv: "2609.24526"
code: https://github.com/MachEmbodied/ME-VLM
original:
  - ../../sources/papers/me_vlm_arxiv_2609_24526.md
  - ../../sources/repos/me_vlm.md
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ./paper-me-brain-1-0.md
  - ./paper-me-u0.md
  - ../overview/li-auto-machembodied-4-papers-technology-map.md
sources:
  - ../../sources/papers/me_vlm_arxiv_2609_24526.md
  - ../../sources/sites/me-vlm.md
  - ../../sources/repos/me_vlm.md
  - ../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md
summary: "ME-VLM（arXiv:2609.24526）：4B/35B-A3B 统一具身认知与 Agent；双专家 RL + 多教师蒸馏；ME-Brain 认知核；代码与权重待发布。"
---

# ME-VLM：统一具身认知与 Agent 协调的 VLM

**ME-VLM**（*A Unified VLM for Embodied Cognition and Agent Coordination*，[arXiv:2609.24526](https://arxiv.org/abs/2609.24526)，[项目页](https://machembodied.com/ME-Brain/ME-VLM.html)，[仓库](https://github.com/MachEmbodied/ME-VLM)）是 **MachEmbodied（理想汽车基础模型团队）** 的统一视觉–语言模型，提供 **4B** 与 **35B-A3B** 两档，在同一建模框架内融合 **具身认知**（几何/时序/执行反馈）与 **多模态 Agent**（规划、工具调用、失败归因与修正）。

## 一句话定义

**一个 VLM 同时做「看懂物理世界里的下一步」和「数字任务里的多轮 Agent」，靠分专家 RL 再蒸馏合并，而不是简单串联规划器与执行器。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ME-VLM | MachEmbodied-VLM | 本文统一 VLM |
| VLM | Vision-Language Model | 视觉–语言多模态骨干 |
| RL | Reinforcement Learning | 具身 / Agent 双专家强化学习 |
| SFT | Supervised Fine-Tuning | 两阶段具身能力注入 |
| MoE | Mixture of Experts | 35B-A3B 稀疏激活变体 |

## 为什么重要

- [ME-Brain 1.0](./paper-me-brain-1-0.md) 的 **Cognitive Core** 与本文训练路线直接对应：Brain 讲系统协同，ME-VLM 讲认知核 **如何练出来**。
- 训练强调 **执行反馈与结果检查**（仿真堆叠案例中认错目标后根据反馈改计划），与「只出计划不验结果」的级联管线对比。
- **开源结论：待发布**（步骤 2.5，2026-09-25）— 技术报告与 PDF 已释；推理/训练/4B·35B 权重与端侧工具包 **TODO**。

## 核心机制

| 阶段 | 内容 |
|------|------|
| **能力注入** | 多模态基座 + 空间关系、 affordance、动作前提/结果；练习 **选下一步** 与 **检查是否完成** |
| **双专家 RL** | 具身专家：物理约束、空间判断、执行反馈；Agent 专家：拆解、工具、多轮交互 |
| **多教师蒸馏** | 学生在 **自身轨迹** 上由专家 on-policy 指导，合并为单一部署模型 |
| **端侧（4B）** | Visual token 压缩、W4A8 量化、M100 NPU 协同（论文报告 prefill 400ms→188ms） |

```mermaid
flowchart TB
  sft[两阶段 SFT\n具身能力注入]
  rl_e[具身专家 RL]
  rl_a[Agent 专家 RL]
  dist[多教师 on-policy 蒸馏]
  deploy[4B / 35B-A3B 部署]
  sft --> rl_e
  sft --> rl_a
  rl_e --> dist
  rl_a --> dist
  dist --> deploy
```

## 实验与评测

| 指标 | 报告值 | 读法 |
|------|--------|------|
| **ME-VLM 35B-A3B 具身基准 avg** | **70.9** | 26 项具身相关基准综合（物理理解、规划、执行、纠错等） |
| **Agent 基准 avg** | **72.5** | 多模态理解、工具、长程规划等套件综合 |
| **仿真堆叠案例** | 定性 | 首轮认错目标 → 反馈后改计划，第三轮完成 |
| **真机** | 未给大样本 SR | 与 ME-Brain 共享系统背景；**勿重复计为第二套独立真机集** |

## 与其他工作对比

| 维度 | ME-VLM | 级联「VLM 规划 + 独立 VLA 执行」 | [ME-U0](./paper-me-u0.md) |
|------|--------|----------------------------------|---------------------------|
| 决策层 | 统一 VLM 内嵌 Agent + 具身认知 | 模块边界清晰但反馈闭环弱 | 理解专家选子任务/接触点 |
| 输出 | 语言/结构化决策 → 下游动作模块 | 同上 | 直接联合生成视觉未来 + 动作 |
| 开源 | 报告已释，权重/代码 TODO | 依具体项目 | **已开源** ME-U0 仓库 |

## 结论

**ME-VLM 是 MachEmbodied 栈的「认知与 Agent 中枢」：用分专家 RL + 蒸馏解决具身与数字任务优化目标冲突，再向 ME-Brain 记忆–动作环供决策。**

1. **待发布** 推理与权重前，第三方只能读报告与项目页案例，无法复现榜单数字。
2. 70.9 / 72.5 为 **多基准平均**，不可与单一 LIBERO/RoboTwin 数字直接横比。
3. 训练强调 **失败恢复与结果检查** — 部署时应保留执行反馈通道，而非开环计划。
4. 4B 端侧路径说明车企场景对 **延迟与量化** 的要求，与云端 35B 形成产品梯度。
5. 与 ME-U0：**VLM 定「做什么/对不对」**；U0 定 **「怎么动 + 世界怎么变」** 的联合生成。

## 源码运行时序图

不适用：核对 `MachEmbodied/ME-VLM@4717f41` 官方仓库 `README.md`，Todo 中 **Inference code / Training code / Pretrained model weights (4B / 35B-A3B) / Edge deployment toolkit** 均为未完成（仓库当前仅含技术报告 PDF）。待官方发布推理入口后，应按「加载 4B/35B → 多模态 prompt + 图像 → 具身/Agent 决策 → 下游执行器」补序图。

## 工程实践

| 项 | 论文披露工程做法 |
|----|-----------------|
| **源码运行时序图** | **不适用**（代码与权重未开源；仅技术报告已发布） |
| **视觉 Token 压缩** | 免训练即插即用 **Max–Min 贪心多样性筛选**，多视角各自按同比例压缩（4,096 → 3,328 tokens），并按原时序索引重排以保时空结构（4B 保留 96.7%、35B 保留 98.2% 分数） |
| **M100 SoC 软硬件协同** | 4B 变体采用 **W4A8 量化**（权重降 75%，精度损失 <1%），由 **Arm CPU 推理运行时 + M100 NPU RISC-V 固件** 协同执行，prefill 延迟由 **400 ms 降至 188 ms（2.13× 加速）** |
| **跨域迁移** | 具身能力可直接轻量适配自动驾驶（MAPLM / CODA-LM / nuScenes-QA）与具身导航（R2R / RxR 闭集离散动作多轮对话接口） |

## 局限与风险

- **代码与权重尚未开源：** 截至 2026-11-21，`MachEmbodied/ME-VLM` 仅放出了技术报告 PDF，4B/35B-A3B 权重、训练脚本与端侧部署工具包均未发布，第三方无法复现榜单数字。
- **部分评测项存在异常空缺：** Table 1 中 4B 模型在 MindCube 与 RoboBench-Error 两项基准录得异常结果（记为 `-`），说明小参数版在特定多视角/错误诊断基准上稳定性仍弱于 35B-A3B。
- **硬件协同指标绑定自研芯片：** 400 ms→188 ms prefill 延迟优化深度依赖理想 M100 SoC（24 核 Arm CPU + LPDDR5X 273 GB/s + M100 编译工具链），在通用桌面 GPU 上的吞吐收益未单列验证。
- **综合均分与真机统计边界：** 70.9 / 72.5 均为多基准平均分且部分基线标注了复测或未报告差异；论文主要给出真机定性复合放置示例，缺乏大样本真机成功率统计。

## 关联页面

- [ME-Brain 1.0](./paper-me-brain-1-0.md)
- [MachEmbodied-U0](./paper-me-u0.md)
- [VLA](../methods/vla.md)
- [四篇技术地图](../overview/li-auto-machembodied-4-papers-technology-map.md)

## 参考来源

- [me_vlm_arxiv_2609_24526.md](../../sources/papers/me_vlm_arxiv_2609_24526.md)
- [wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md](../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)
- [arXiv:2609.24526](https://arxiv.org/abs/2609.24526)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.24526)
- [ME-VLM 项目页](https://machembodied.com/ME-Brain/ME-VLM.html)
- [ME-Brain 总览](https://machembodied.com/index.html#brain)
