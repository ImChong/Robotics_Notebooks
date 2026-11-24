---
type: entity
project_id: nopeworld
arxiv: "2611.27963"
project: https://nopeworld.github.io/
code: https://github.com/yuhangdai427/NoPeWorld
tags: [paper, world-models, video-generation, physical-consistency, polyu]
status: draft
updated: 2026-11-24
related:
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
  - ../../roadmap/depth-wam.md
sources:
  - ../../sources/papers/nopeworld_arxiv_2611_27963.md
  - ../../sources/sites/nopeworld.md
  - ../../sources/repos/nopeworld.md
original:
  - ../../sources/papers/nopeworld_arxiv_2611_27963.md
summary: "冻结视频世界模型，从 DiT 隐状态预测候选相机动作是否违反几何约束，在线替换动作以减少障碍穿透。"
---

# NoPeWorld：非穿透世界模型（Non-penetrating World Models）

## 一句话定义

冻结视频世界模型，从 DiT 隐状态预测候选相机动作是否违反几何约束，在线替换动作以减少障碍穿透。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| DiT | Diffusion Transformer | 冻结的视频去噪骨干 |
| VLM | Vision-Language Model | 开放域视频评审 |
| SIS | Safe-Intent Score | 安全且尽量保持意图的评估指标 |

## 为什么重要

冻结视频生成器的物理约束与动作筛选边界。将其作为可比较的独立研究或工程入口，而不是仅记录热度。

## 核心信息

| 字段 | 内容 |
|---|---|
| 机构 | 香港理工大学 Learning & Generation Lab |
| 论文 | [arXiv:2611.27963](https://arxiv.org/abs/2611.27963) |
| 日期 | 2026-11-20 |
| 开放状态 | Apache-2.0；2026-11-21 发布代码、六种生成器的预测头与基准数据；生成器本体须另行获取。 |

## 核心原理与流程总览

训练阶段以 Unreal 场景真实几何标注候选运动安全性，冻结生成器，仅学习不足 200 万参数的头。每个 chunk 评估 34 个候选，包含原动作、停止、缩短步幅与转向。预测模式读取上一 chunk 的状态；碰撞模式读当前 chunk 首次去噪状态。保留安全原动作，否则选择尽量保持意图的候选，再闭环生成视频。

以下为已查阅官方资料支持的概念流程，完整原文核对待补。

```mermaid
flowchart TD
    A["真实几何与候选运动"]
    B["安全标签"]
    C["冻结 DiT 隐状态与小预测头"]
    D["候选动作安全评分"]
    E["保留或替换命令"]
    F["下一视频 chunk"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

## 源码运行时序图

`nopeworld/cli.py::infer_main`、`scripts/infer.py`；已核对 CLI 分派，rollout 内部调用链尚待完整阅读。官方命令 `nopebench-download --what weights`、`nopeworld-infer --generator hy_worldplay --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/hy/collision`。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户 CLI
    participant C as nopeworld.cli.infer_main
    participant R as inference.rollout.run
    U->>C: nopeworld-infer --generator hy_worldplay --method nopeworld_collision --task task2
    C->>C: set_roots / generator_config / method_available
    C->>R: run(generator, method, task, out, routes, group, seeds)
    R-->>U: 路线生成输出
```

## 工程实践

Apache-2.0；2026-11-21 发布代码、六种生成器的预测头与基准数据；生成器本体须另行获取。 复现入口与版本依赖见参考来源。本次未执行训练、推理或仿真复现。

## 实验与评测

NoPeBench Task 1 在真实历史上测试决策，Task 2 在已知几何的闭环生成中评估，Task 3 用开放域静态及移动障碍图像测试零样本迁移。作者报告六模型 Task 2 碰撞模式违规率平均 74%→16%；LingBot-World 2.0 89%→7%。预测模式不是所有模型均支持，Evoke-Turbo 无跨 chunk 记忆。

## 结论

冻结视频世界模型，从 DiT 隐状态预测候选相机动作是否违反几何约束，在线替换动作以减少障碍穿透。 工程选型时应同时验证指标口径及使用边界。

- 区分真实历史决策、闭环视频违规和开放域评审。
- 同时监测误干预与意图保留，不能只优化停止率。
- 将该方法当生成器侧动作过滤，而非真机安全控制器。

## 局限与风险

视频不穿透不等于机器人安全；几何评估依赖重建和分割，开放域依赖 VLM 判断；预测头并未提供形式化安全保证。安装文档要求至少 80 GB 显存，部分模型建议 96 GB。

**核对状态：** 草稿；完整原文及必要源码尚未通读，不宣称健康度满分。

## 关联页面

- [generative-world-models](../methods/generative-world-models.md)
- [world-action-models](../concepts/world-action-models.md)
- [相关纵深路线](../../roadmap/depth-wam.md)

## 参考来源

- [papers：nopeworld](../../sources/papers/nopeworld_arxiv_2611_27963.md)
- [sites：nopeworld](../../sources/sites/nopeworld.md)
- [repos：nopeworld](../../sources/repos/nopeworld.md)

## 推荐继续阅读

- [论文 HTML](https://arxiv.org/html/2611.27963)
- [arXiv](https://arxiv.org/abs/2611.27963)
- [官方项目页](https://nopeworld.github.io/)
- [官方源码](https://github.com/yuhangdai427/NoPeWorld)
- [预测头权重](https://huggingface.co/PeterDAI/NoPeWorld)
- [NoPeBench 数据](https://huggingface.co/datasets/PeterDAI/NoPeBench)
