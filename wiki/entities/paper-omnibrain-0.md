---
type: entity
project_id: omnibrain-0
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-11-06
arxiv: "2610.36013"
project: https://public-bots.github.io/OmniBrain-0/
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
original:
  - ../../sources/papers/omnibrain_0_arxiv_2610_36013.md
sources:
  - ../../sources/papers/omnibrain_0_arxiv_2610_36013.md
  - ../../sources/sites/omnibrain-0.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "OmniBrain-0（arXiv:2610.36013）：通过具身预训练、领域专家蒸馏和多教师在策略蒸馏整合导航与操作决策。"
---

# OmniBrain-0（arXiv:2610.36013）

**OmniBrain-0**（*OmniBrain-0: A Unified Agentic Model for Embodied Intelligence*，[arXiv:2610.36013](https://arxiv.org/abs/2610.36013)，[项目页](https://public-bots.github.io/OmniBrain-0/)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「VLA 与策略后训练」组。

## 一句话定义

**通过具身预训练、领域专家蒸馏和多教师在策略蒸馏整合导航与操作决策。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MOPD | Multi-teacher On-Policy Distillation | 多教师在策略蒸馏 |
| RSI | Recursive Self-Improvement | 递归自我改进 |
| VLA | Vision-Language-Action | 视觉语言动作策略 |

## 为什么重要

- 把导航与操作统一到决策模型。再把执行证据变为评分规则和验证轨迹，考察长任务恢复，收益仍需结合具体工具接口理解。
- 开源结论：**模型代码待发布**（2026-11-06）：[官方仓库](https://github.com/luo-yc17/OmniBrain-0) README 标注 Code: coming soon；公开仓库包括项目展示资源。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    subgraph RT["推理：统一 Agent 闭环（§3）"]
        G["任务指令 g + 观测 o_t（图像、本体状态、任务状态）+ 历史 h_t + 可选经验库 M"] --> AG["OmniBrain-0（Qwen3.5-9B）选择工具 k_t 与参数 ξ_t"]
        AG --> TL["工具：感知 / VLA 操作 / VLN 导航 / 运动控制"]
        TL --> FB["新观测 o_t+1 + 执行反馈 f_t+1"]
        FB --> G
    end
    subgraph TRN["三阶段训练（§4）"]
        DP["交互数据构建：归一化、质量控制、推理重构、一致性验证<br/>保留 canonical 成功与 recovery 轨迹"] --> BASE["Embodied 预训练（具身 VQA + 长程交互数据）→ OmniBrain-Base"]
        BASE --> TM["域专家蒸馏：操作 / 导航 teacher（监督式 hard distillation）"]
        DP --> TM
        TM --> MOPD["MOPD：对学生生成的前缀做 masked reverse-KL，仅更新学生"]
        BASE --> MOPD
    end
    MOPD --> AG
    subgraph RSI["Rubric-Guided RSI 数据飞轮（§5）"]
        V["Verifier 评估成败与效率"] --> MA["Meta-Agent 分析证据，增改 rubrics"]
        MA --> RM["Reminder policy：检测到问题时在工具调用前插入提醒（参数不变）"]
        MA --> VT["筛选并验证轨迹，扩充域专家语料"]
    end
    AG -.交互轨迹.-> V
    RM -.step 级提醒.-> G
    VT --> TM
```

图依据：论文 §3 / §4 / §5 / Fig.3–7

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.36013](https://arxiv.org/abs/2610.36013) |
| **开源** | 模型代码待发布；项目展示资源公开 |
| **要点** | 通过具身预训练、领域专家蒸馏和多教师在策略蒸馏整合导航与操作决策。 |
| **文内指标** | LIBERO-Pro 9B 为 65.5%，RSI 为 69.3%；具体评测口径见下节。 |

## 源码运行时序图

**不适用：** 官方 README 标注模型代码待发布；当前 scripts 仅列展示素材准备脚本，未见模型训练或推理入口。

## 工程实践

- 接口需分别适配感知、动作、导航与反馈；每次调用后根据新观测更新状态。主表的 OmniBrain 结果不使用外部经验库。
- 训练使用 16 张 NVIDIA H20；论文中导航采用 SmartWay，操作调用 π0.5、LingBot-VLA 或 RLDX-1 等技能。官方模型代码仍待发布。

## 局限

- 成功率依赖底层技能覆盖与可靠性；未见任务组合、空间定位仍有差距。真机仅覆盖三类任务、每类 20 次。
- 论文正文与附录对真机物体颜色描述不一致，复现时需向作者确认；R2R-CE 智能体结果仅基于 100 个 Val-Unseen 样本。

## 实验与评测

项目页列出的 9B / RSI 成功率：LIBERO-Pro 65.5% / 69.3%；LIBERO-Plus 83.8% / 84.5%；RoboCasa365 39.6% / 43.2%；RoboTwin 2.0 88.2% / 89.4%；R2R-CE 65.0% / 66.0%。R2R-CE 为 Val-Unseen 的 rand100 子集；不宜与完整集结果直接比较。

**阅读状态：** 已阅读 arXiv v1 HTML 正文及附录文字；图内细节仍以 PDF 原图为准。

- 实验口径：LIBERO-Pro 800 次、LIBERO-Plus 840 次、RoboTwin 2.0 500 次；RoboCasa365 为 50 任务宏平均，不同基准的 Overall 不可直接等同。
- 真机三任务宏平均 73.3%，π0.5 为 58.3%，教师为 86.7%；小样本结果不能推广到所有实物任务。
- 消融：去掉恢复轨迹并仅保留单条成功轨迹时，LIBERO-Pro 成功率下降约 9%；该试验同时改变了数据组成。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把导航与操作统一到决策模型：再把执行证据变为评分规则和验证轨迹，考察长任务恢复，收益仍需结合具体工具接口理解。**

1. 开源边界：项目展示资源公开，模型代码待发布。
2. 核心机制：通过具身预训练、领域专家蒸馏和多教师在策略蒸馏整合导航与操作决策。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [omnibrain_0_arxiv_2610_36013.md](../../sources/papers/omnibrain_0_arxiv_2610_36013.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.36013](https://arxiv.org/abs/2610.36013)
- [arXiv v1 HTML](https://arxiv.org/html/2610.36013v1)：已读正文及附录文字，HTML 中部分图仅有图注。

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.36013)
- [项目页](https://public-bots.github.io/OmniBrain-0/)
