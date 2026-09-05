# 具身智能走到世界模型这一步，“问题”反而更多了

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能走到世界模型这一步，“问题”反而更多了
- **副标题：** 机器人究竟需要怎样的“世界观”？
- **类型：** blog
- **作者：** 李海伦；编辑徐青阳（腾讯科技）
- **原始链接：** https://mp.weixin.qq.com/s/2DEpiexjwh5O6bBJDXk3LA
- **发表日期：** 2026-09-05
- **入库日期：** 2026-09-05
- **抓取方式：** iPhone UA curl（Jina / 桌面 UA 常返回微信「环境异常」）
- **原始抓取落盘：** [`sources/raw/wechat_tencent_world_model_questions_2026-09-05.md`](../raw/wechat_tencent_world_model_questions_2026-09-05.md)
- **一句话说明：** 腾讯科技把「世界模型」过载写成路线分化：Atlas / 功能分类、LeCun 族 LpWM→LeVJEPA、光象 Phi-WM 1.0 ActEffect；文内点名论文均落独立 `paper-*` 节点，已有页不重复造。

## 核心摘录（归纳，非全文）

文内判断：VLA 之后世界模型、隐空间、RL/触觉同时上场，术语比路线更先过载。席宁（港大）称仍处百花齐放。真正拉开差距的是「模型从数据里学会了什么」，不是再堆成功示范。

### 文内点名 → 本库节点

| # | 资料 | 身份 | 开源结论（入库日） | wiki |
|---|------|------|-------------------|------|
| 01 | Atlas（World Labs，2026-09-01） | 产品 / omni 模型，非论文 | **确认未开源**（早期访问） | [atlas-world-model](../../wiki/entities/atlas-world-model.md) **复用** |
| 02 | A Functional Taxonomy of World Models（Fei-Fei，2026-06） | 博客 / 概念文 | **确认未开源** | [functional-taxonomy-world-models](../../wiki/concepts/functional-taxonomy-world-models.md) **复用**（#1808）；原文 [World Labs 博客](https://www.worldlabs.ai/blog/taxonomy-of-world-models) |
| 03 | LeJEPA（arXiv:2511.08544） | 论文（LeVJEPA 前身） | **已开源** CC BY-NC 4.0 | [paper-lejepa](../../wiki/entities/paper-lejepa.md) **新建** |
| 04 | LeWM / LeWorldModel（arXiv:2603.19312） | 论文（LpWM 稠密对照） | **已开源** MIT + HF | [paper-lewm](../../wiki/entities/paper-lewm.md) **新建** |
| 05 | LpWM（arXiv:2608.22764，2026-08-24） | 论文 | **已开源** MIT | [paper-lpwm](../../wiki/entities/paper-lpwm.md) **新建** |
| 06 | LeVJEPA（arXiv:2608.27395，2026-08-27） | 论文 | **已开源** MIT + 权重 NC | [paper-levjepa](../../wiki/entities/paper-levjepa.md) **复用** |
| 07 | Phi-WM 1.0 ActEffect（光象 / 清华，技术报告） | 技术报告 | **确认未开源**（仅 PDF） | [paper-phi-wm-acteffect](../../wiki/entities/paper-phi-wm-acteffect.md) **新建** |

### 文内要点速记

1. **功能三分：** Renderer / Simulator / Planner 是同一 POMDP 环的投影，不是三家互斥产品。
2. **LpWM：** 稀疏非负隐变量替代 LeWM 稠密高斯，动作后状态更好预测，规划器更简单。
3. **LeVJEPA：** 把 LeJEPA 拉到视频，少算力学运动与前后状态。
4. **ActEffect：** 世界模型当**训练时反馈器**，部署蒸馏进策略一次前向；显隐状态解耦；仿真反事实是数据主力。
5. **商业：** Phi-Bot X1 先把汽车产线的「1」跑通；仿真资产可信度是工程瓶颈。

## 对 wiki 的映射

- **7/7 独立详情节点**（产品 / 概念复用 3，论文新建 4）；**0 重复 arXiv**。
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[WAM](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 点名论文独立节点核查（新建 4 / 复用 3 / **0 重复 arXiv**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
