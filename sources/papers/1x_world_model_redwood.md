# 1X World Model（1XWM / Redwood AI 评测引擎）

> 来源归档（paper / tech report）

- **标题：** 1X World Model: Evaluating Bits, not Atoms（补充技术进展报告；产品叙事亦称 Redwood AI World Model）
- **类型：** paper / company tech report
- **机构：** 1X Technologies
- **项目/博文：** <https://www.1x.tech/discover/redwood-ai-world-model>（2025-06-16）
- **技术报告 PDF：** <https://www.1x.tech/1x-world-model.pdf>
- **相关开源挑战：** <https://github.com/1x-technologies/1xgpt>（World Model Challenge 数据集与 GENIE 基线；**不等于** 本报告所述完整 1XWM 评测引擎）
- **入库日期：** 2026-07-26
- **一句话说明：** 面向全身人形（NEO）的动作条件视频世界模型：预测未来观测并输出任务级 state-value，用作策略/checkpoint **离线评测引擎**；强调动作可控性、与真机评测相关性，以及自治 rollout 数据的缩放。

## 核心摘录（MVP）

### 1) 动机：用 bits 评测，而不是用 atoms 穷举真机

- **链接：** <https://www.1x.tech/1x-world-model.pdf>
- **摘录要点：** 家庭场景评测一个 checkpoint 往往需要成百上千次采样，一轮训练又产生大量候选；真机评测吞吐成为瓶颈。1XWM 在相同初始观测下对多条低层动作序列生成未来并打分，用于架构对比、checkpoint 选择、长尾生产失败集的反事实回放。
- **对 wiki 的映射：**
  - [1XWM 实体页](../../wiki/entities/paper-1xwm-redwood-world-model.md)
  - [WAM×运动控制五路径](../../wiki/overview/wam-motion-control-five-paths.md) — ⑤ 策略评估外侧

### 2) 方法：视觉/动作编码 → 骨干 → 视频解码 + state-value 头

- **链接：** <https://www.1x.tech/discover/redwood-ai-world-model>
- **摘录要点：** 输入视频帧、机器人观测与动作轨迹，编码到 latent 后预测未来帧 latent，并预测终帧任务成功/完成价值。博文强调：动作可控性（同起点多动作反事实）、自治策略 rollout 数据缩放、跨任务迁移、以及 WM 排序与真机排序的相关性（例：proprioception 消融、ViT-L vs 其他编码器对照）。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md)
  - [1X Technologies](../../wiki/entities/1x-technologies.md)

### 3) 开源边界（步骤 2.5）

- **部分 / 相关开源：** 技术报告与博文描述的 **完整 1XWM 评测栈未作为单一官方训练仓完整发布**。公开相关资源：
  - 技术报告 PDF（方法与评测叙事）
  - `1x-technologies/1xgpt`：World Model Challenge（EVE 第一人称 token 数据 + GENIE 基线）
  - HF `1x-technologies/worldmodel`、`worldmodel_raw_data` 等数据集
- **勿等同：** Challenge 基线 ≠ 报告中 Redwood 生产评测引擎。

## 关键术语

- **1XWM：** 1X World Model
- **Redwood AI：** 1X 策略/模型产品叙事名；世界模型博文挂在 Redwood 发现页下
- **State value：** 对终态任务成败/完成度的预测头，用于策略排序

## 关联 Wiki 页面

- [paper-1xwm-redwood-world-model](../../wiki/entities/paper-1xwm-redwood-world-model.md)
- [1x-technologies](../../wiki/entities/1x-technologies.md)
- [wam-motion-control-five-paths](../../wiki/overview/wam-motion-control-five-paths.md)

## 当前提炼状态

- [x] 博文 + 12 页技术报告摘要
- [x] 与 1xgpt Challenge 边界写清
- [x] wiki 映射
