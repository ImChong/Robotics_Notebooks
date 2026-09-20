---
type: entity
tags: [paper, llm, 3d, manipulation, stanford, nvidia]
status: complete
updated: 2026-09-20
arxiv: "2307.05973"
code: https://github.com/huangwl18/VoxPoser
related:
  - ./paper-pai-2209-07753-codeaspolicies.md
  - ./paper-omnimanip.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/wechat_lumina_embodied_practice_part2_code_as_policy_2026-09-20.md
summary: "VoxPoser（arXiv:2307.05973）：LLM/VLM 把语言约束转为 3D value map，再交 motion planner 生成轨迹；huangwl18/VoxPoser 已开源。"
---

# VoxPoser

**VoxPoser**（[arXiv:2307.05973](https://arxiv.org/abs/2307.05973)，[代码](https://github.com/huangwl18/VoxPoser)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**不直接输出关节角，而是让 LLM 在体素空间里「画」出该去哪、该避哪。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| LLM | Large Language Model | 大语言模型 |
| IL | Imitation Learning | 模仿学习 |
| BC | Behavior Cloning | 行为克隆 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | Stanford / NVIDIA 等 |
| **arXiv** | [2307.05973](https://arxiv.org/abs/2307.05973) |
| **开源** | **已开源** |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明「轻接触 / 稳定 ≠ 高任务成功率（见 HRC benchmark 对照）」——回原文与该 benchmark 须核对：**过程指标（接触力、稳定性）与结果指标（完成率）可能背离**，只报其一会得出相反判断；官方仓可复现仿真管线。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与端到端 VLA 对照：本文可解释、可加约束，但完成率常低于端到端；本页结论另点明与 Code-as-Policies 对照「代码原语 vs 体素场」两种中间表示 |
| **横比口径** | 过程指标与结果指标分属两类，横比前须声明用的是哪一类；且仿真管线成绩不等于真机成绩。 |
| **开源状态** | **已开源** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

VoxPoser 代表「3D 中间表示 + 传统规划器」路线：可解释、可约束，但完成率常低于端到端 VLA。

- value map 是规划接口，不是最终策略
- 轻接触/稳定 ≠ 高任务成功率（见 HRC benchmark 对照）
- 官方 GitHub 可复现仿真管线
- 与 CaP 对照：代码原语 vs 体素场

## 源码运行时序图

官方仓库提供训练/推理入口；节点对齐 README。

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Repo as 官方仓库
    participant Policy as 策略
    participant Env as 仿真/真机
    Dev->>Repo: clone + 依赖
    Dev->>Policy: 加载权重
    loop 控制环
        Env->>Policy: 观测
        Policy->>Env: 动作
    end
```

## 关联页面

- [paper-pai-2209-07753-codeaspolicies](./paper-pai-2209-07753-codeaspolicies.md)
- [paper-omnimanip](./paper-omnimanip.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [wechat_lumina_embodied_practice_part2_code_as_policy_2026-09-20.md](../../sources/blogs/wechat_lumina_embodied_practice_part2_code_as_policy_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2307.05973](https://arxiv.org/abs/2307.05973)
