---
type: entity
project_id: motion-mamba-2
arxiv: "2611.29897"
project: https://aigeeksgroup.github.io/MotionMamba2.0/
code: https://github.com/AIGeeksGroup/MotionMamba2.0
tags: [motion-generation, humanoid, diffusion, pku, ntu]
status: draft
updated: 2026-11-25
summary: "以根轨迹和身体动作的双阶段扩散生成配合循环状态及持久内容记忆，支持变化指令下的流式参考运动生成。"
related:
  - ./paper-halo-perceptive-humanoid-motion-sim.md
  - ./paper-embo-harness.md
  - ../../roadmap/depth-motion-generation.md
sources:
  - ../../sources/papers/motion-mamba-2.md
  - ../../sources/sites/motion-mamba-2.md
  - ../../sources/repos/motion-mamba-2.md
original:
  - ../../sources/papers/motion-mamba-2.md
---

# Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control｜长程人形控制的自回归扩散 Mamba

## 一句话定义

以根轨迹和身体动作的双阶段扩散生成配合循环状态及持久内容记忆，支持变化指令下的流式参考运动生成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| PCM | Persistent Content Memory | 持久内容记忆 |
| CFG | Classifier-Free Guidance | 无分类器引导 |
| SSM | State Space Model | 状态空间模型 |

## 核心信息

| 项目 | 内容 |
|---|---|
| 机构 | 北京大学、高丽大学、蒙纳士大学、南洋理工大学 |
| 发布 | 2026-11-23（v1） |
| 原文 | [arXiv:2611.29897](https://arxiv.org/abs/2611.29897) |
| 项目 | [主页](https://aigeeksgroup.github.io/MotionMamba2.0/) |
| 源码 | [GitHub](https://github.com/AIGeeksGroup/MotionMamba2.0) |

## 核心原理

每个窗口为 52 帧、25 FPS；文本、空间目标和已接受历史共同约束根轨迹去噪，根轨迹局部特征再条件化身体去噪。局部注意力读取近期特征，Mamba 循环状态跨窗恢复，PCM 保存可供注意力检索的摘要。去噪过程中历史保持固定，仅接受干净窗口后提交历史状态。生成参考运动与 SONIC 跟踪控制是不同层次。

## 流程总览

```mermaid
flowchart TD
  A["文本与空间目标"] --> B["根轨迹扩散"]
  H["已接受历史：KV、Mamba、PCM"] --> B
  H --> C["身体扩散"]
  B --> C
  C --> D["接受窗口并更新历史"]
  D --> H
  D --> E["SONIC 跟踪"]
```

## 源码运行时序图

```mermaid
sequenceDiagram
  participant U as scripts.simulation
  participant L as visualization.launch
  participant P as plans
  participant S as 六个服务进程
  U->>L: main
  L->>L: environment / check / prepare
  L->>P: env, inference spec
  P-->>L: text, generator, web, physics, policy, viewer
  L->>S: 启动服务（入口级）
```

`python -m scripts.simulation start` → `scripts/simulation.py` → `visualization/launch.py`；已读启动器的 environment/check/prepare/plans，六服务为 text、generator、web、physics、policy、viewer。训练入口 `python -m scripts.training_motion_mamba2`，评测 `python -m scripts.evaluate`；未执行。 此图仅覆盖已核对入口，内部完整调用关系待复核。

## 评测与实验

论文 §V–VII：HiPHI 988 帧（39.52 s）条件下 Root CE 7.55 cm、End CE 13.64 cm；RTX 4090 的 52 帧完整窗口 p50 由 1685.35 ms 降至 276.14 ms，配对中位加速 6.138 倍。此为稳态窗口生成而非端到端响应延迟。G1 真机 100 次试验覆盖四种任务；用户提供的 94/100 与无摔倒统计仍待逐项核对表格。

## 结论

当前证据支持将此工作作为相应方法的工程参考，而非通用真机性能保证。

- 将生成器和真机跟踪器分别验收。
- 以长程终端误差和窗口延迟同时评估。
- 区分稳态推理与用户命令响应。

## 局限与风险

论文明确暂未覆盖物体操作；生成器不等价于低层运控。项目页列有代码、模型和数据；外部 SONIC、LLM2Vec 及资产仍受各自许可和安装条件约束。

> 草稿：原文与必要源码未全部通读，尚未运行仓库健康度脚本或 preflight。

## 关联页面

- [paper-halo-perceptive-humanoid-motion-sim](./paper-halo-perceptive-humanoid-motion-sim.md)
- [paper-embo-harness](./paper-embo-harness.md)
- [纵深路线](../../roadmap/depth-motion-generation.md)

## 推荐继续阅读

- [官方项目](https://aigeeksgroup.github.io/MotionMamba2.0/)

## 参考来源

- [论文归档](../../sources/papers/motion-mamba-2.md)
- [项目页归档](../../sources/sites/motion-mamba-2.md)
- [源码归档](../../sources/repos/motion-mamba-2.md)
