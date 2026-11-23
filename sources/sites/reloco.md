# ReLoco — 官方项目主页（2026-11-21 核对）

> 来源归档（site / academic project）

- **名称：** ReLoco（Recovery and Locomotion）
- **标题：** ReLoco: Unified Humanoid Locomotion and Fall Recovery via Behavior-Conditioned Motion Priors
- **中文：** 行走、跌倒自主起身、继续执行原速度指令的一体化人形控制
- **项目页：** https://lequn-f.github.io/ReLoco/
- **论文：** https://arxiv.org/abs/2611.23916（v1，2026-11-18）
- **项目演示：** 项目页提供 MuJoCo 交互体验，按 8/2、4/6、7/9 调节速度与转向，P 触发推扰，0 重置
- **代码/权重/数据：** 截至 2026-11-21，在主页公开的链接区与页脚**未找到**独立代码仓库、可下载权重或数据；不能将在线试玩误称“代码已开源”
- **核对范围：** 项目页标题、作者机构、交互 demo 说明、Abstract、Method、reference refinement、真实机器人实验与页脚链接，交叉核对 arXiv v1。
- **通读：** 已完成
- **入库日期：** 2026-11-21
- **wiki 映射：** [ReLoco 独立实体](../../wiki/entities/paper-reloco-unified-locomotion-fall-recovery.md)

## 官方项目页可验证的内容

- 一个 proprioceptive actor 执行 stand/walk/run、俯卧/仰卧/侧卧起身及恢复原命令；训练中由姿态选择 locomotion/get-up expert pool，双输出共享判别器；判别器、参考追踪器、辅助力、路由在部署侧都不需要。
- 真实平台是北京人形机器人创新中心的 **Tiangong 3.0**，论文量化为 1.70 m、64.5 kg；项目展示正面/背面扰动后的恢复及多种表面行走。
- **来源边界：** 真机成功率和保护事件使用论文 Table VII 的完整统计；在线 MuJoCo demo 只是一种浏览器体验，不等同于提供训练/推理源码，也不能据此推断模型许可证或复现条件。

## 继续阅读

- [论文完整归档](../papers/reloco_arxiv_2611_23916.md)
- [论文 HTML 原文](https://arxiv.org/html/2611.23916)
- [ReLoco 项目主页](https://lequn-f.github.io/ReLoco/)
