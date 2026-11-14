---
type: entity
project_id: starvla-framework
arxiv: "2604.05014"
project: https://starvla.github.io/
code: https://github.com/starVLA/starVLA
code_scope: "starVLA_dev 主仓的通用架构/数据加载/训练/评测基础设施"
project_distinction: "独立 framework 论文 arXiv:2604.05014；不同于既有 StarVLA-alpha arXiv:2604.11757 的简化策略对照研究"
tags: [paper, vla, framework, modular-training, benchmarks, reproducibility, hkust]
status: complete
updated: 2026-11-14
related:
  - ../methods/star-vla.md
  - ../methods/vla.md
  - ../comparisons/vla-open-source-ecosystem-2026-three-shifts.md
  - ./paper-openwam.md
original:
  - ../../sources/papers/starvla_framework_arxiv_2604_05014_full_reading.md
sources:
  - ../../sources/papers/starvla_framework_arxiv_2604_05014_full_reading.md
  - ../../sources/repos/star_vla.md
summary: "StarVLA 框架论文（arXiv:2604.05014）提供通用 VLM/世界模型骨干与 FAST/OFT/flow/GR00T 动作头可换插模块、跨本体与多目标训练和统一 server-client benchmark 接口；与 StarVLA-alpha（2604.11757）是两篇不同论文。"
---

# StarVLA：模块化 VLA 实验与部署框架 / A Lego-like Codebase for Vision-Language-Action Model Developing

## 一句话定义

**StarVLA** 是把 **backbone、action head、dataset adapter、trainer、evaluation server** 拆开的开源 VLA 研究平台，使同一实验管线可以切换 VLM/世界模型骨干及动作生成策略，比较结构而非重复维护不同项目代码。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作的通用策略研究框架 |
| VLM | Vision-Language Model | 可被替换的视觉语言骨干 |
| WAM | World-Action Model | 支持视频世界模型骨干的同生态方向 |
| FAST | Fast Action Sequence Tokenization | 自回归离散动作头选项 |
| OFT | OpenVLA Optimized Fine-Tuning | 本框架的并行连续回归动作头族 |
| SFT | Supervised Fine-Tuning | 可复用的监督模仿训练 |
| EEF | End Effector | 策略输出接口的重要物理量 |

## 为什么重要

StarVLA 的核心价值不是“又一个 VLA 成功率最高”，而是能够让研究者在统一观测/动作协议下，控制地替换模型骨干、解码头、异构数据组合和 benchmark，以此更可信地研究哪个部件提高了性能。

## 核心原理

- **组合架构：** 支持 Qwen3-VL 等 VLM 与 Cosmos 等世界模型相关骨干；动作头覆盖 StarVLA-FAST、OFT、PI/flow 和 GR00T 双系统。以相同输入输出接口替换组件。
- **训练：** 共享 SFT、多模态协同训练和跨本体数据混合策略，数据加载器返回模型无关原始字典，具体框架负责预处理与 `forward/predict_action`。
- **评测：** 服务端加载策略，客户端接入 LIBERO/RoboTwin/SimplerEnv/RoboCasa 等基准；同一框架下的任务/数据/参数仍须固定以确保可比。
- **独立论文身份：** 此页对应 StarVLA **框架论文 2604.05014**；[StarVLA-α 方法页](../methods/star-vla.md) 对应 **2604.11757**，侧重极简动作头控制变量实验，不能混作同一论文。

## 流程总览

```mermaid
flowchart TD
  dataset["多源机器人数据/多模态语料"] --> loader["可插拔 dataloader 和数据适配"]
  model["可替换 VLM / 世界模型 backbone"] --> fw["共享 framework 接口"]
  head["FAST / OFT / PI / GR00T 动作头"] --> fw
  loader --> fw
  fw --> trainer["Trainer / 多任务数据配比"]
  trainer --> ckpt["Checkpoint"]
  ckpt --> server["Policy server"]
  sim["LIBERO / RoboTwin / SimplerEnv 评测客户端"] <--> server
```

## 源码运行时序图

官方 `docs/starVLA_guideline.md` 与 `README` 提供的 LIBERO 路径；可核验 `run_libero_train.sh` 使用 `accelerate launch` / ZeRO-2，评测为两终端 server/client。

```mermaid
sequenceDiagram
  autonumber
  participant U as 用户
  participant SH as run_libero_train.sh
  participant A as accelerate + DeepSpeed
  participant D as LeRobot dataloader
  participant F as StarVLA framework
  participant P as Policy server
  participant E as LIBERO client
  U->>SH: 配好 starvla_cotrain_libero.yaml 启动
  SH->>A: accelerate launch 配置 ZeRO-2
  A->>D: 加载 data_mix 数据集
  A->>F: 按 framework.name 初始化 backbone 和动作头
  D-->>F: image、lang、action、state
  F-->>A: loss 与 checkpoint
  U->>P: 加载 checkpoint 运行 policy server
  E->>P: 发送 obs + task
  P->>F: predict_action
  F-->>E: 连续动作块
```

## 实验与评测

- 论文覆盖 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、跨 benchmark 联合训练和扩展性测试；成功率是指定策略版本与评测环境的作者报告。
- LIBERO 示例协议每套 10 tasks × 每任务 50 rollouts；正文比较 30k 训练步与 OpenVLA-OFT 175k 步，**不同训练轮数、配置或数据不可假定完全等价**。
- 官方 README 的训练计划已支持 SFT、多模态协同与跨本体 co-training；**RL adaptation 尚列为待办**。不要因论文泛谈 RL 就声称所有算法都可直接重现。

## 结论

**StarVLA 提供能够独立替换 VLA 结构和训练配方的实验基础设施，不应将其当作单一固定动作网络。**

1. 做模型消融时先固定数据、骨干、动作头以外的变量。
2. 将训练脚本与评测 server/client 协议分别检查。
3. 星标、仓库支持列表和论文成功率应标清版本/协议。
4. 对 RL 训练路线优先核对实际可运行入口，不能把 TODO 当 release。
5. StarVLA 框架论文与 StarVLA-α benchmark 论文分别维护。

## 与其他工作对比

- [StarVLA-α 方法详解](../methods/star-vla.md)：框架研究 vs 受控极简基线研究的不同论文。
- [OpenWAM](./paper-openwam.md)：后者深挖世界–动作联合生成预训练变量；StarVLA 定位更广的 VLA 实验组合。
- [2026 VLA 三条路线](../comparisons/vla-open-source-ecosystem-2026-three-shifts.md)：属于工程基础设施。

## 工程实践

官方仓库公开 `starVLA_dev` 分支与 `docs/starVLA_guideline.md`、`examples/simBenchmarks/LIBERO/train_files/` 和 benchmark server/client 工具。建议用示例 checkpoint 先跑 smoke test 与单套 LIBERO，再改骨干和数据混合。此处为源码静态核对，未做 GPU 复现。

## 局限与风险

不同模型头可能有不同 tokenizer、动作归一化与显存预算；“统一框架”并不使所有 checkpoint / benchmark 完全可互换。论文介绍的训练扩展吞吐也不是相同硬件上的策略推理频率。

## 关联页面

- [StarVLA-α](../methods/star-vla.md)
- [VLA](../methods/vla.md)
- [2026 VLA 项目对比](../comparisons/vla-open-source-ecosystem-2026-three-shifts.md)

## 参考来源

- [StarVLA 框架论文原文核对](../../sources/papers/starvla_framework_arxiv_2604_05014_full_reading.md)
- [StarVLA 源码索引](../../sources/repos/star_vla.md)

## 推荐继续阅读

- [arXiv:2604.05014](https://arxiv.org/html/2604.05014)
- [GitHub](https://github.com/starVLA/starVLA)
- [StarVLA-α: arXiv 2604.11757](https://arxiv.org/abs/2604.11757)
