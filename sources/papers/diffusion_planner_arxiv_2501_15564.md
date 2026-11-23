# Diffusion-Based Planning for Autonomous Driving with Flexible Guidance（Diffusion Planner，arXiv:2501.15564v2）

> 原始论文归档；中文：基于扩散模型与灵活引导的自动驾驶闭环规划。

- **类型：** paper / autonomous-driving / diffusion / motion-planning / imitation-learning
- **标题：** Diffusion-Based Planning for Autonomous Driving with Flexible Guidance
- **作者：** Yinan Zheng、Ruiming Liang、Kexin Zheng、Jinliang Zheng、Liyuan Mao、Jianxiong Li、Weihao Gu、Rui Ai、Shengbo Eben Li、Xianyuan Zhan、Jingjing Liu
- **机构：** 清华大学、中国科学院自动化研究所、香港中文大学、上海交通大学、毫末智行（HAOMO.AI）、上海人工智能实验室
- **会议：** ICLR 2025 Oral
- **论文：** <https://arxiv.org/abs/2501.15564>；[v2 HTML 全文](https://arxiv.org/html/2501.15564v2)；[PDF](https://arxiv.org/pdf/2501.15564)
- **版本：** arXiv v1 2025-01-26，v2 2025-02-09
- **项目主页：** <https://zhengyinan-air.github.io/Diffusion-Planner/>
- **代码：** <https://github.com/ZhengYinan-AIR/Diffusion-Planner>
- **权重：** <https://huggingface.co/ZhengYinan2001/Diffusion-Planner>
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv v2 HTML 的 §1–6、相关工作、方法 Eq. (4)–(8)、nuPlan/配送车评测 Table 1–3、消融、附录 A–E（含 Table 4–6、推理超参、引导能量、局限），并与官方项目页/README 的开放状态和复现入口对照。
- **通读：** 已完成
- **一句话说明：** 联合生成自车与周车未来轨迹的 DiT 扩散闭环规划器；借由可微能量的推理时引导调整驾驶偏好，纯模型输出在 nuPlan Test14 (R) 为 82.93，Test14-hard (R) 为 69.22。

## 摘录 1：问题与核心贡献（§1、§4）

- 传统模仿规划难表达多模态驾驶与交互，常依靠输出后的规则筛选/修正；本文把自车规划与邻车预测统一为条件轨迹去噪。
- 输入为当前状态、周车历史、矢量化车道/静态对象与导航路线；用独立 MLP-Mixer 编码路线与场景，Transformer 融合后，DiT 的 MHCA / AdaLN 将场景、路线、扩散时刻注入去噪网络。
- 训练监督是联合未来轨迹的重建误差；预测状态包括平面位置与航向的 sin/cos，之后由下游车辆控制器跟踪，**不是直接输出油门/刹车**。
- **对 wiki 的映射：** [独立实体](../../wiki/entities/paper-diffusion-planner-autonomous-driving.md)；与 [Diffusion Policy](../../wiki/entities/paper-diffusion-policy.md) 和 [自动驾驶技术地图](../../wiki/overview/autonomous-driving-core-algorithms-series.md) 做机制/接口对照。

## 摘录 2：无训练额外分类器的引导与运行（§4.3–4.4、附录 C）

- 使用扩散 posterior sampling 近似，在每轮采样时通过可微能量 \(E(\hat x_0)\) 对去噪 score 做梯度引导；能量项可组合：速度目标、jerk 舒适度、车与车碰撞、可行驶区域。
- **引导免重训≠没有手工设计约束**：能量函数需人为定义且可微；「无需规则后处理」指**直接输出主评测**不调用输出后的 rule-based refinement，不应泛化为无规则/可验证安全证明。
- 数据增强：初始车辆状态加扰动、未来轨迹五次插值、ego-centric 坐标与 z-score 归一化。
- 附录 Table 5：预测 10 辆邻车、decoder 3 blocks、hidden 192、6 attention heads、10 个去噪步；DPM-Solver++ 低温采样；单张 A6000 上约 20 Hz（附录称 8 s/10 Hz horizon 约 0.05 s）。
- **对 wiki 的映射：** 在实体页说明流程、公式直觉与 [源码运行时序图](../../wiki/entities/paper-diffusion-planner-autonomous-driving.md)。

## 摘录 3：结果与边界（Table 1–4、附录 D–E）

| 数据集/设置 | Diffusion Planner 不经 refine | 典型对照 | 需要注意 |
|---|---:|---:|---|
| nuPlan Test14 (Reactive) | 82.93 | PlanTF 79.58；PDM-Closed 91.63 | 学习型纯输出领先，不是所有类型绝对最高 |
| nuPlan Test14-hard (Reactive) | 69.22 | PlanTF 61.61；PDM-Closed 75.19 | 不应与非 Reactive 分数混比 |
| nuPlan Test14 (Non-reactive) | 89.19 | PLUTO w/o refine 89.90 | 此列 PLUTO 数字更高 |
| 配送车闭环（约 200 h 自采数据） | 92.08 | PlanTF 90.89；PLUTO w/o refine 87.77 | 原数据集尚未公开，第三方复核受限 |

- 加上 **额外 rule-based refinement** 后，Test14 Reactive 91.75、Test14-hard Reactive 82.00；这些**不能算作无后处理能力**。
- 局限：依赖矢量地图和车检测；大横向车道变换和未覆盖的行为分布表现弱；输出轨迹与下游控制器存在执行 gap；扩散多次采样仍耗计算。
- 官方 README 将训练/仿真/引导与模型权重列为已开放；配送车 200 h 数据集仍列待发布（审批中）。
- **对 wiki 的映射：** 记录闭环指标和官方开放边界，不把论文实验成绩写成真机量产安全保证。

## 一手资料与映射

- [v2 论文全文](https://arxiv.org/html/2501.15564v2)
- [ICLR 2025 OpenReview](https://openreview.net/forum?id=wM2sfVgMDH)
- [官方项目页](../sites/diffusion-planner.md)
- [官方代码归档](../repos/diffusion-planner.md)
- [论文实体节点](../../wiki/entities/paper-diffusion-planner-autonomous-driving.md)
