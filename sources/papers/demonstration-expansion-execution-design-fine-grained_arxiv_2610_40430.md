# 示范扩展与执行设计（arXiv:2610.40430）

> 来源归档（paper）

- **标题：** Improving Fine-Grained Manipulation through Demonstration Expansion and Execution Design
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40430>
- **PDF：** <https://arxiv.org/pdf/2610.40430>
- **作者：** Gu, Jingxu, Lian, Dongze
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 通过示范扩展与执行设计改善精细机器人操作。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 精细操作不仅依赖策略预测，还受到示范覆盖范围、相机变化、动作执行周期和夹爪控制映射的影响。
3. **周报所述创新点：** 提出 DEED，对同一物理轨迹生成不同语言指令、相机位置和光照条件下的训练示范。部署时结合逆运动学、任务专用夹爪映射和闭环动作块重规划，在 MetaFine 五项任务评测中取得 85.55% 的总体评分。
4. **arXiv 摘要（英文节选）：** Fine-grained manipulation requires a robot policy to follow instructions, remain robust to visual variations, and translate predicted actions into precise robot control. We present Demonstration Expansion and Execution Design (DEED), a framework for adapting a pretrained vision-language-action policy to the five MetaFine tasks through structured demonstration expansion and execution design. DEED generates demonstrations for different instructions from the same initial scene and renders each physical trajectory under different camera positions and lighting conditions while preserving its original states and action targets.

**对 wiki 的映射**

- [paper-demonstration-expansion-execution-design-fine-grained](../../wiki/entities/paper-demonstration-expansion-execution-design-fine-grained.md)
