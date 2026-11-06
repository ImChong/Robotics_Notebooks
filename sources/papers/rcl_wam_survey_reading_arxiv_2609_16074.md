# RCL WAM 综述论文通读（技术地图视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/rcl-awesome-wam-technology-map.md`）

- **标题：** World-Action Models for Robot Learning and Control: A Survey
- **作者：** Zuxing Lu、Hongjia Zhai 等（MBZUAI 等），通讯 Xingxing Zuo
- **类型：** paper / survey / world-action-models
- **原文：** <https://arxiv.org/abs/2609.16074>（v1，2026-09-13；本次通过 arXiv PDF 获取全文并转文本阅读）
- **一句话说明：** 机器人视角的 WAM 综述，给出 WAM 与 WM / VLA 的形式化区分、联合预测 vs 逆动力学两类转移建模、端到端 vs 双系统两类架构、预训练→后训练管线与数据金字塔；本技术地图是其配套 Awesome 清单的站内展开。
- **核对范围：** 通读摘要、§I 引言、§II 背景（定义与公式 1–15）、§III 架构与分类（组件、转移范式、架构结构、训练管线、数据模态与规模）、§VI 开放挑战、§VII 结论。§IV 应用（操纵 / 导航 / 驾驶）、§V 数据集与评测表格、参考文献仅浏览标题与结构，未逐条核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **三类模型的形式化（§II-A）：** WM：ẑ_{t+1}=f_WM(z_t,a_t)；VLA：â_{t:t+H}=f_VLA(o_{<t},ℓ)，不显式建模未来状态；WAM：(ô_{t:t+H}, â_{t:t+H})=f_WAM(o_{<t},ℓ)，联合预测未来观测与动作块，显式建模动作如何影响未来状态。
- **四个生成组件（§II-C）：** 策略 p(a|o,ℓ)、前向动力学 p(o_future|o,a)、逆动力学 p(a|o,o_future)、视觉规划 p(o_future|o,ℓ)；动作生成两条路径：联合建模 p_joint(o,a|…)，或 plan-then-act：p_plan(ô|…)·p_idm(a|ô,o)。
- **四条基础线索（§II-B/C）：** MBRL（动力学用于规划）、视频生成模型 VGM（未来场景预测）、LAPA 式潜在动作（从无动作视频学）、VLA（视觉语言到动作）。
- **转移建模（§III-B）：** 联合预测（GR-1/GR-2、VPP、UWM、WorldVLA、Motus、Cosmos Policy 等）与逆动力学（Seer、DreamZero、LingBot-VA、Fast-WAM、GigaWorld-Policy 等）；作者指出两者无普遍优劣，是不同权衡。
- **架构结构（§III-C）：** 端到端（单一骨干处理全部模态）vs 双系统（视频专家与动作专家分离，经 MoT 共享自注意力或 cross-attention 连接，如 π0 / GR00T / Helix）；与转移范式正交，构成 2×2。
- **训练管线（§III-D）：** 预训练含时空表征（Wan / Cosmos 等视频骨干）、动作表征（FAST、LAPA、离散 token vs 连续）、视频-动作对齐（UWM 的四条对齐路线）；后训练含策略微调（LoRA、适配器、残差动作头；DreamZero 约 30 分钟真机数据实现跨本体迁移）、数据增广（world-model 重渲染与 IDM 伪动作标注）、强化学习（仿真 RL / 真实 RL / 世界模型想象 RL 三路线）。
- **数据金字塔（§III-E，图 7）：** 互联网视频（量大、提供动力学先验）→ 第一视角人类示范 → 具身轨迹（量小、提供动作落地）；Dyna-2 把第一视角视频扩到百万小时，作为 WAM 可从无动作标注视频学习的例证。
- **开放挑战（§VI）：** 运动意图与动作对齐（保留预训练先验）、世界-动作因式分解（联合优化可能产生过于乐观的状态被策略利用）、空间与多视角一致性、长程记忆、神经模拟器与闭环策略学习、推理延迟（少步生成、量化 / 蒸馏 / KV cache）。

## 对 wiki 的映射

- [`wiki/overview/rcl-awesome-wam-technology-map.md`](../../wiki/overview/rcl-awesome-wam-technology-map.md)
- [`wiki/entities/paper-rcl-wam-robot-learning-control-survey.md`](../../wiki/entities/paper-rcl-wam-robot-learning-control-survey.md)
- [`wiki/concepts/world-action-models.md`](../../wiki/concepts/world-action-models.md)
