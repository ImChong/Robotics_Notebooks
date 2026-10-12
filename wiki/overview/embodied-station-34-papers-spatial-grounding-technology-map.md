---
type: overview
tags: [overview, survey, vla, world-model, manipulation, benchmark, technology-map]
status: complete
updated: 2026-10-12
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md
summary: "具身智能小站 2026-10-12 的 34 篇盘点：策略部署与时间建模、空间指令诊断、世界–动作模型、经验筛选与微调、灵巧与全身控制、导航记忆与评测；每篇一个独立详情节点。"
---

# 空间指令、时间建模与世界模型：34 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 34 篇盘点](https://mp.weixin.qq.com/s/HSsx-QeViQjmnsOTL_FU2Q)（2026-10-12）提供阅读坐标，并把文章中的 34 条论文逐条指向独立详情节点。文章本身按“1 篇深读 / 2 篇跟进 / 31 篇扫读”排序，本页保留该顺序。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 以视觉、语言条件生成机器人动作的策略 |
| WAM | World-Action Model | 联合预测环境变化与机器人动作的模型路线 |
| JEPA | Joint-Embedding Predictive Architecture | 在潜空间预测未来表示的自监督架构，LeWAM 以此联合学习世界与动作 |

## 为什么重要

- 文章的主线是“会操作不等于听得懂空间指令”：Spa-Bench 用真机配对场景把操作能力与空间理解拆开诊断，Chronos 则从“物理时间”角度处理部署时的延迟与异步。
- 34 篇里既有已开源的代码库，也有大量“待发布”或只有匿名镜像的条目；各详情节点逐一核对了项目页 / 仓库，开源状态以节点为准，不要以文章里“代码：”字样为准。
- 多个节点记录了论文摘要与表格数字不一致、HTML 附录未读到等核实边界，复现前请回到原文。

## 阅读路线

```mermaid
flowchart LR
  A["先读: 物理时间 Chronos"] --> B["跟进: 空间诊断 Spa-Bench / 联合世界动作 LeWAM"] --> C["扫读: 31 篇按方向回看"]
```

## 34 篇独立论文详情

| # | 论文节点 | arXiv | 文章中的一句话判断 |
|---|----------|-------|--------------------|
| 01 | [Chronos Policy：在物理时间上学习机器人策略](../entities/paper-chronos-policy-physical-time.md) | [2610.13929](https://arxiv.org/abs/2610.13929) | 让机器人策略显式理解物理时间 |
| 02 | [Spa-Bench：真机上检验 VLA 的空间落地](../entities/paper-spa-bench-spatial-grounding-vla.md) | [2610.12696](https://arxiv.org/abs/2610.12696) | 会操作，不代表理解空间指令 |
| 03 | [LeWAM：用同一个潜在模型执行与规划](../entities/paper-lewam.md) | [2610.12818](https://arxiv.org/abs/2610.12818) | 用同一个潜在模型执行与规划 |
| 04 | [Evo-RL：从有选择的经验中迭代改进真机策略](../entities/paper-evo-rl.md) | [2610.13345](https://arxiv.org/abs/2610.13345) | 用有价值的经验持续改进机器人策略 |
| 05 | [SmolWAM：把世界-动作模型压到边缘设备](../entities/paper-smolwam.md) | [2610.13740](https://arxiv.org/abs/2610.13740) | 把世界动作模型压缩到更低推理成本 |
| 06 | [RARA：检索锚定的表征对齐微调](../entities/paper-rara-retrieval-anchored-finetuning.md) | [2610.13578](https://arxiv.org/abs/2610.13578) | 少量示范微调时保住原有泛化能力 |
| 07 | [Human2Dex：腕部视角人类示范驱动多只灵巧手](../entities/paper-human2dex-wrist-centric-dexterous.md) | [2610.13119](https://arxiv.org/abs/2610.13119) | 同一份人类示范服务不同灵巧手 |
| 08 | [FDM：动作到未来的前向动力学策略](../entities/paper-fdm-forward-dynamics-models.md) | [2610.13117](https://arxiv.org/abs/2610.13117) | 用未来监督动作，部署时省去未来生成 |
| 09 | [JitterVLA：VLA 人机操作的运动质量回放基准](../entities/paper-jittervla-replay-benchmark.md) | [2610.13628](https://arxiv.org/abs/2610.13628) | 在同一命令流上比较机器人运动平滑性 |
| 10 | [Grounded-Dreamer：带物理验证的世界模型合成示范](../entities/paper-grounded-dreamer.md) | [2610.13164](https://arxiv.org/abs/2610.13164) | 用物理验证筛选世界模型合成示范 |
| 11 | [Evo-World：用光流监督的轻量 VLA 未来运动预测](../entities/paper-evo-world-optical-flow-vla.md) | [2610.12936](https://arxiv.org/abs/2610.12936) | 用光流监督轻量策略的未来运动预测 |
| 12 | [EASG：把动作坐标系画进图像，让 VLM 直接当动作模型](../entities/paper-easg-vlm-native-action-model.md) | [2610.12832](https://arxiv.org/abs/2610.12832) | 让视觉语言模型直接看懂动作坐标系 |
| 13 | [Fly-by-Code：让空中操作的编码智能体主动查明失败原因](../entities/paper-fly-by-code-aerial-manipulation.md) | [2610.12549](https://arxiv.org/abs/2610.12549) | 让飞行操作智能体主动查明失败原因 |
| 14 | [PanTact：用可穿戴接触接口教人形机器人全身操作](../entities/paper-pantact-whole-body-contact-interface.md) | [2610.12558](https://arxiv.org/abs/2610.12558) | 让人形机器人用身体接触完成操作 |
| 15 | [ECL：证据引导的反事实学习，缓解 VLA 长尾任务被头部动作压制](../entities/paper-ecl-evidence-guided-counterfactual-long-tail.md) | [2610.12861](https://arxiv.org/abs/2610.12861) | 减轻长尾任务被高频动作模式压制 |
| 16 | [AutoBio+：带动态实验现象的生物实验室灵巧操作基准](../entities/paper-autobio-plus-lab-dexterous-benchmark.md) | [2610.13643](https://arxiv.org/abs/2610.13643) | 把实验室动态现象纳入机器人评测 |
| 17 | [Causal Diffusion Planner：移动机器人精确推放多个物体](../entities/paper-causal-diffusion-nonprehensile-multi-object-placement.md) | [2610.12684](https://arxiv.org/abs/2610.12684) | 用因果扩散规划精准推放多个物体 |
| 18 | [REMM：给冻结操作策略加“可修订记忆”](../entities/paper-remm-revisable-embodied-memory.md) | [2610.12710](https://arxiv.org/abs/2610.12710) | 扰动后撤回过时结论，再安排修复 |
| 19 | [HF-WM：向人类交互学习并建模失败动力学的世界模型](../entities/paper-hf-wm-failure-dynamics-world-model.md) | [2610.13522](https://arxiv.org/abs/2610.13522) | 世界模型也需要学习失败的交互动力学 |
| 20 | [Robot-WATCHDOG：用对象关系图检测操作失败](../entities/paper-robot-watchdog-object-graph-failure-detection.md) | [2610.13403](https://arxiv.org/abs/2610.13403) | 用对象关系图识别操作失败 |
| 21 | [CASTER：把生成视频当作约束优化的初值](../entities/paper-caster.md) | [2610.12567](https://arxiv.org/abs/2610.12567) | 把生成视频转为受约束的机器人轨迹 |
| 22 | [TAME-Nav：时序非对称 MoE 蒸馏的敏捷导航](../entities/paper-tame-nav.md) | [2610.12877](https://arxiv.org/abs/2610.12877) | 通过时序专家蒸馏适应复杂导航场景 |
| 23 | [StereoEngine：没有双目相机也能扩充双目示范](../entities/paper-stereoengine.md) | [2610.12734](https://arxiv.org/abs/2610.12734) | 没有双目相机，也能扩充双目示范 |
| 24 | [PriMoRetarget：用优先级层级保住关键接触的人形动作重定向](../entities/paper-primoretarget.md) | [2610.12642](https://arxiv.org/abs/2610.12642) | 用优先级保住动作迁移中的关键接触 |
| 25 | [AlphaNIST：只用二元奖励学会 NIST 装配](../entities/paper-alphanist.md) | [2610.13354](https://arxiv.org/abs/2610.13354) | 只用二元奖励学习接触丰富的装配 |
| 26 | [MeMoREval：具身智能体该记文字、图像还是场景图？](../entities/paper-memoreval-embodied-memory-modality.md) | [2610.12561](https://arxiv.org/abs/2610.12561) | 比较机器人该存文字、图像还是场景图 |
| 27 | [AFCE：让 VLA 预测「动作将产生的效果」而不只是动作](../entities/paper-afce-action-future-coupled-effect-vla.md) | [2610.13212](https://arxiv.org/abs/2610.13212) | 让策略预测动作将产生的物理效果 |
| 28 | [BOPI：按新旧信念的一致程度修订人形推车计划](../entities/paper-bopi-belief-overlap-partial-inversion.md) | [2610.13087](https://arxiv.org/abs/2610.13087) | 根据新旧信念的一致程度修改推车计划 |
| 29 | [MARGoS：把群体强化学习的复现变成平台职责](../entities/paper-margos-reproducible-marl-argos.md) | [2610.13772](https://arxiv.org/abs/2610.13772) | 把群体强化学习复现变成平台职责 |
| 30 | [ATHLETE：用通用行走数据加普通视频学人形反应式网球](../entities/paper-athlete-reactive-humanoid-tennis-tgmm.md) | [2610.13572](https://arxiv.org/abs/2610.13572) | 从普通视频学习反应式人形网球 |
| 31 | [LeapBot-JAC：世界-动作模型里的两条联合动作链](../entities/paper-leapbot-jac-joint-action-chains.md) | [2610.12886](https://arxiv.org/abs/2610.12886) | 在固定动作词元预算内安排两条动作链 |
| 32 | [e-puck 通信价值演化：信号只在带来新信息时出现](../entities/paper-epuck-value-of-information-communication.md) | [2610.13118](https://arxiv.org/abs/2610.13118) | 通信何时对群体机器人真正有用 |
| 33 | [MIRAGE：达芬奇控制台里的统一手术导航与可视化框架](../entities/paper-mirage-davinci-surgical-navigation.md) | [2610.12780](https://arxiv.org/abs/2610.12780) | 把多模态手术导航集成到机器人控制台 |
| 34 | [事件相机动作识别：多帧表示上的 MAE 预训练](../entities/paper-event-action-recognition-multi-frame-mae.md) | [2610.13856](https://arxiv.org/abs/2610.13856) | 用多帧预训练识别长时段事件动作 |

## 关联页面

- [Manipulation 任务页](../tasks/manipulation.md)
- [VLA 方法页](../methods/vla.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [具身智能小站 · 上一期盘点（协作数据与世界模型 12 篇）](collab-wm-12-papers-technology-map.md)

## 参考来源

- [具身智能小站 · 34 篇盘点 来源归档](../../sources/blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/HSsx-QeViQjmnsOTL_FU2Q)

## 推荐继续阅读

- [arXiv cs.RO 最新列表](https://arxiv.org/list/cs.RO/recent)
