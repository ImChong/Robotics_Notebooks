# DAWN（arXiv:2610.40537）

> 来源归档（paper）

- **标题：** DAWN: Diffusion-based Alignment and unfastening Workflow Network
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40537>
- **PDF：** <https://arxiv.org/pdf/2610.40537>
- **作者：** Kumar, Hemant, Goenka, Piyush, Chopra, Nikhil
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** DAWN：基于扩散策略的对准与螺钉拆卸流程网络。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 自动螺钉拆卸需要完成精确对准、工具插入和持续施力，多个阶段的误差会相互影响。
3. **周报所述创新点：** 将任务分为对准插入和拆卸两个阶段。使用 Diffusion Policy 完成工具对准，利用多模态分类器判断阶段切换，再通过力控制完成螺钉拆卸，在 Franka 机器人上验证了完整流程。
4. **arXiv 摘要（英文节选）：** We present DAWN (Diffusion-based Alignment and unfastening Workflow Network), a hierarchical robotic framework designed to perform real-world, unfastening tasks through multi-modal sensory integration. Autonomous robotic unfastening is crucial for improving recycling processes, enhancing safety, and reducing operational costs; yet, it remains predominantly manual due to inherent challenges in precise screwdriver engagement with fasteners, detailed force execution, and robustness to environmental variability. To address the complexity of learning long-horizon, contact-rich manipulation tasks, we subdivide the unfastening task procedure into two sequential stages: (1) alignment & insertion, and (2) unfastening.

**对 wiki 的映射**

- [paper-dawn-diffusion-alignment-unfastening](../../wiki/entities/paper-dawn-diffusion-alignment-unfastening.md)
