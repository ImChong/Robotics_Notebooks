---
type: comparison
tags: [embodied-ai, vla, world-action-model, humanoid, whole-body-control, sim2real]
status: complete
updated: 2026-09-28
related:
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../concepts/embodied-three-layer-control-architecture.md
  - ../overview/vla-evolution-lineage.md
  - ../overview/wam-motion-control-five-paths.md
sources:
  - ../../sources/sites/robot-foundation-model-company-research-2026.md
summary: "按世界/动作基础模型、通用人形整机、强全身控制三种阅读视角，对照 12 家团队的公开技术路线与复现边界。"
---

# 机器人基础模型与通用人形：公司技术路线对照（2026）

## 一句话定义

把公司技术分享按**世界与动作建模、整机层级控制、全身技能与仿真迁移**三种阅读视角组织，可以更快找到训练目标、动作接口和真机闭环的不同答案；一家公司可以同时出现在几条路线中。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 由视觉和语言条件生成机器人动作 |
| WAM | World-Action Model | 联合或级联建模世界后果与动作的模型族；须逐项目检验推理接口 |
| WBC | Whole-Body Control | 全身运动与接触约束的协调控制 |
| RTC | Real-Time Action Chunking | 在执行中衔接新旧动作块以减小推理停顿 |
| RL | Reinforcement Learning | 用交互反馈优化策略 |

## 三种视角与各自的核心问题

| 阅读视角 | 代表团队和作品 | 阅读时追问 |
| --- | --- | --- |
| 世界与动作基础模型 / VLA | [PI](../../sources/sites/pi-website-technical-articles.md)：π₀→π₀.₇、FAST、RTC、MEM；1X：World Model / Redwood；Galbot：AstraBrain-WAM；Galaxea：G0 / Fast-WAM；[AgiBot](../../sources/sites/agibot-world.md)：GO / AgiBot World；Google DeepMind：Gemini Robotics | 预测的是未来观测、未来动作还是二者？推理时真的运行世界预测吗？数据、权重和训练代码开放到哪一层？ |
| 通用人形整机与大小脑接口 | Figure：Helix→Helix 02→Helix 2.5；[Skild](../entities/skild-ai.md)：Skild Brain；[NVIDIA](../entities/isaac-gr00t.md)：GR00T / Cosmos / Isaac Lab；LimX：COSA / FluxVLA；1X：NEO / Redwood；Unitree：G1 / UnifoLM | 视觉语言模块输出关节目标、运动 latent 还是高层指令？高频低层由谁执行？不同模块延迟如何闭环？ |
| 强全身技能、Real2Sim2Real | [Light Origins](../entities/light-o1.md)：Light-O1、REACT、Parkour、Nav；Galbot：AstraBrain-WBC；LimX：腿足技能 / COSA；Unitree：G1 控制生态；NVIDIA：GR00T Control / Isaac Lab | 人视频/动作先验如何变成可执行参考？仿真如何覆盖接触、跌倒、损伤？真机部署观察和动作频率是什么？ |

上述是**学习视角**，并非互斥的公司赛道：例如 Light-O1 也做视觉语言动作预训练，1X 既研究世界模型又构建整机。

## 四个具体对照

1. **π 系与 1X 世界模型：** π 系公开了 VLA 动作生成、动作块执行与记忆的多个独立研究问题；1X 世界模型主要展示对动作条件下未来视频/行为的建模。不能把“有世界模型”直接推导为“已公开端到端 World Model→Policy 的部署实现”。
2. **Figure 与 NVIDIA：** Helix 发布聚焦在 Figure 机器人上的多系统全身闭环；NVIDIA 公开 Cosmos、仿真、GR00T 和端侧平台等多层资产。学习 Figure 时追系统接口，学习 NVIDIA 时追数据生成、训练到部署的实际模块边界。
3. **Light Origins 与 Galbot：** 前者的公开材料适合追人类动作预训练、Real2Sim2Real 和韧性控制；后者区分 AstraBrain-WAM 与 AstraBrain-WBC，适合观察“世界侧”和“执行侧”如何连接。WBC 也不自动等于端到端 WAM。
4. **开源程度：** PI 的 openpi、Light-O1 的推理代码/预览权重、NVIDIA 的 Isaac-GR00T 等提供具体复现入口；Figure / Skild 的公开博客主要提供方法与实验叙述。比较时应以各项目资源页为准，不用公司是否有 GitHub 账号代替资产核查。

## 建议阅读顺序

- **想研究 VLA / Flow / chunk：** [PI 逐篇索引](../../sources/sites/pi-website-technical-articles.md) → [VLA 演进](../overview/vla-evolution-lineage.md) → [VLA 纵深](../../roadmap/depth-vla.md)。
- **想研究世界预测如何帮助执行：** [1X World Model 归档](../../sources/sites/1x-world-model-redwood.md) → [WAM 概念](../concepts/world-action-models.md) → [WAM 纵深](../../roadmap/depth-wam.md)；再比 Galbot / Galaxea 的最新发布。
- **想研究人形全身落地：** [Light-O1](../entities/light-o1.md) → [具身三层控制架构](../concepts/embodied-three-layer-control-architecture.md) → [全身运控技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)；对照 Figure Helix 和 GR00T。
- **想动手复现：** 优先核对官方仓库里的**代码、权重、数据、真机接口**是否齐备；按 [VLA 开源复现谱系](../overview/vla-open-source-repro-landscape-2025.md) 选与硬件匹配的项目。

## 局限与风险

截至 2026-09-28，此页比较的是**官方公开材料及已收录资料**，并非统一基准实验。不同团队的任务、本体、频率和测评环境不同，不能从宣传演示直接排出性能名次。1X 的世界模型、Figure 的 Helix 与各 WAM 的源码开放范围应按单篇项目页复核。

## 关联页面

- [WAM 概念与分类](../concepts/world-action-models.md)
- [VLA 方法总览](../methods/vla.md)
- [VLA 演进](../overview/vla-evolution-lineage.md)
- [具身三层控制架构](../concepts/embodied-three-layer-control-architecture.md)
- [人形运控小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)

## 参考来源

- [12 家公司官方技术入口与开放程度索引](../../sources/sites/robot-foundation-model-company-research-2026.md)
- [PI 官方技术文章逐篇索引](../../sources/sites/pi-website-technical-articles.md)
- [1X World Model / Redwood 项目归档](../../sources/sites/1x-world-model-redwood.md)
- [Light-O1 项目页及开源核查](../../sources/sites/light-o1.md)
- [Skild AI 官方站归档](../../sources/sites/skild-ai.md)

## 推荐继续阅读

- [Figure Helix 02 官方技术文章](https://www.figure.ai/news/helix-02)
- [Light Origins 官方技术博客](https://www.lightorigins.com/en/blog/)
- [NVIDIA Robotics Blog](https://developer.nvidia.com/blog/tag/robotics/)
