# X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving（arXiv:2603.19979）

> 来源归档（ingest）

- **标题：** X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving
- **缩写：** **X-World**
- **类型：** paper / generative-world-models / autonomous-driving / multi-camera / video-diffusion
- **arXiv：** <https://arxiv.org/abs/2603.19979>（PDF：<https://arxiv.org/pdf/2603.19979>）
- **项目页：** <https://x-world-1.github.io/>
- **代码：** 截至 2026-07-21 项目页仅链论文 PDF，**未列 GitHub / 权重**
- **机构：** 小鹏（XPeng）GWM 团队
- **作者（前若干）：** Chaoda Zheng, Sean Li, Jinhao Deng, Zhennan Wang, Shijia Chen, Liqiang Xiao, Ziheng Chi, Hongbin Lin 等
- **入库日期：** 2026-07-21
- **一句话说明：** **动作条件** 的 **7 摄自车中心生成式世界模型**：给定多视角历史与未来动作序列，在视频空间预测未来环视观测，并支持交通参与者/静态要素与天气等文本外观控制，服务端到端智驾可扩展评测。

## 摘录 1：问题与动机

- **痛点：** 端到端 VLA 智驾仍重度依赖真车路测，成本高、场景覆盖偏、难复现。
- **目标：** 真实世界级模拟器——在提议动作下生成逼真未来观测，且长视界可控、稳定。

**对 wiki 的映射：** [`wiki/entities/paper-x-world.md`](../../wiki/entities/paper-x-world.md)；挂到 [生成式世界模型](../../wiki/methods/generative-world-models.md)「视频即仿真 / 驾驶」分支。

## 摘录 2：核心能力与机制

- **输入/输出：** 同步多视角摄像头历史 + 未来动作 → 未来多摄视频流（严格跟随动作）。
- **控制接口：** 可选动态交通参与者与静态道路要素控制；文本 prompt 做外观级控制（天气、时段、地区风格）；支持在保持动作/场景动力学下做视频风格迁移。
- **骨干：** 多视角潜空间视频生成器，显式鼓励跨视角几何一致性与多样控制下的时序连贯。
- **项目页叙事：** 7 摄 360°「一起看」、连续几十秒不崩、可指挥自车与他车。

**对 wiki 的映射：** 与 [X-Cache](../../wiki/entities/paper-x-cache.md)（推理加速）、[X-Foresight](../../wiki/entities/paper-x-foresight.md)/[X-Mind](../../wiki/entities/paper-x-mind.md)（VLA 内嵌 PWM）形成 **仿真底座 → 加速 → 策略耦合** 链。

## 摘录 3：结果与开源状态

- **宣称性质：** 高质量多视角生成；强跨视角一致、长 rollout 稳定、高可控性（动作跟随与场景控制）。
- **开源：** 截至入库日 **未开源**（项目页无代码链接）。

**对 wiki 的映射：** 实体页写明复现边界；交叉 [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) 概念。
