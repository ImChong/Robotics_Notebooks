# ReflexArc 官方项目主页

> 来源归档（site）；本文仅归纳官方页面和公开资源状态。

- **标题：** ReflexArc: Bottom-Up Integration of Instantaneous Tactile Reflexes into Whole-Body Humanoid Control
- **类型：** site
- **链接：** https://minusmodulo.github.io/reflexarc-website/
- **arXiv：** https://arxiv.org/abs/2611.04424
- **论文全文阅读记录：** [sources/papers/reflexarc_arxiv_2611_04424.md](../papers/reflexarc_arxiv_2611_04424.md)
- **入库日期：** 2026-11-04
- **本次核查日期：** 2026-11-12
- **一句话说明：** 一个直通电机的瞬时触觉撤离通路，与持续执行任务和维持平衡的全身学习策略协同工作。
- **沉淀到 wiki：** [ReflexArc](../../wiki/entities/paper-reflexarc.md)

## 开源状态核查（2026-11-12）

- **代码：待发布。** 官方首页资源区域显示「Code Coming soon」，截至核查时没有可访问的 ARC 官方 GitHub 代码仓库链接，也未展示训练、仿真或真机部署入口。
- **论文：已公开。** arXiv v1 可读，官方项目页有论文入口；顶部部分按钮显示「Paper Coming soon」属于网页界面文案，不代表 arXiv 论文不可读，以实际 arXiv 访问为准。
- **数据/权重：未提供明确下载入口。** 页面展示视频与方法/消融内容，但未核实公开的 ARC-Gym 训练数据、动作先验、模型权重或触觉标定资产。不能据此声称「已开源」。
- **交互演示：可访问，但不等于开源。** 官网的「Interactive trained policy」可供浏览器内体验，并不提供 ARC 代码仓库、模型权重或触觉驱动的下载链接；统计表中的开源状态仍按代码/工件入口判定。
- **源码时序图：不适用。** 只有原论文方法流程，不能冒充代码运行轨迹。

## 页面核对内容

1. **Whole-body reflexes：** 手、臂、躯干、膝与足部触觉的动作展示；接触即刻局部撤离，随后协调全身避让。
2. **Instantaneous tactile reflexes：** 16×8 cm 的 128 taxel 压阻贴片，300 Hz；首帧接触检测和 29 DoF 全身局部二次规划，求解 <4 ms、触觉到电机 <10 ms。
3. **Bottom-up integration：** reflex-in-the-loop、reflex-transparent alignment、reflex-informed adaptation 三个模块；对应 arXiv §IV。
4. **Contact impact：** 延迟 20 ms 的受控对照中，力峰 21.64→14.25 N、冲量 0.2632→0.1225 N·s；对应 arXiv §V-A。
5. **Ablation：** 页面展示 Table I 的部分数据和可视化；定量解释应以原论文 Table I 指标定义为准，勿与箱体冲量实验混淆。
6. **适用边界：** 触觉贴片覆盖有限；未确认作者已发布可复现源码、训练数据、完整真机触觉硬件设计。

## 推荐原始资料

- [arXiv HTML v1](https://arxiv.org/html/2611.04424v1)
- [官方项目主页](https://minusmodulo.github.io/reflexarc-website/)
