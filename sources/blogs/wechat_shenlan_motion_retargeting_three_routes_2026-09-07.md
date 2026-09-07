# 斩获 ICRA 2026 双最佳论文的运动重定向，三条技术路线一次讲透

> 来源归档（blog / 微信公众号 · 深蓝具身智能）

- **标题：** 斩获 ICRA 2026 双最佳论文的运动重定向，三条技术路线一次讲透
- **类型：** blog / wechat / survey / motion-retargeting
- **作者：** 深蓝学院-具身君（编辑｜咖啡鱼；审编｜具身君）
- **原始链接：** https://mp.weixin.qq.com/s/QKPp9grbgpy6-NBNm5Nl-w
- **入库日期：** 2026-09-07
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_motion_retargeting_three_routes_2026-09-07.md`](../raw/wechat_shenlan_motion_retargeting_three_routes_2026-09-07.md)
- **一句话说明：** 综述运动重定向三条技术谱系（IK/优化、深度学习、物理交互感知）及 OmniRetarget 等代表工作；文内每个论文/项目在 wiki 均有**独立**实体页互链，本归档只做索引。

## 核心摘录（按技术路线）

### 路线 1：IK 约束优化

| 主体 | 要点 | wiki |
|------|------|------|
| Retargeting Matters / GMR | 差异化局部缩放、实时多机种、Retargeting Matters 命题 | [Retargeting Matters](../../wiki/entities/paper-hrl-stack-01-retargeting_matters.md) · [GMR 方法页](../../wiki/methods/motion-retargeting-gmr.md) |
| PHC | SMPL 姿态拟合 + 梯度下降；形态差大时偏差大，多作基准 | [PHC](../../wiki/entities/phc.md) |

### 路线 2：深度学习数据驱动

| 主体 | 要点 | wiki |
|------|------|------|
| Human2Humanoid | 骨架感知 GCN + 末端/限位约束；训练与遥操作 | [human2humanoid](../../wiki/entities/human2humanoid.md) · [Learning Human-to-Humanoid](../../wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md) |
| MoReFlow | VQ-VAE token + 流匹配无配对跨角色对齐 | [MoReFlow](../../wiki/entities/paper-moreflow-motion-retargeting-flow.md) |
| AdaMorph | 统一 Transformer 大模型；意图 latent + AdaLN 形态调制 | [AdaMorph](../../wiki/entities/paper-adamorph-unified-motion-retargeting.md) |

### 路线 3：物理约束与交互感知

| 主体 | 要点 | wiki |
|------|------|------|
| DynaRetarget | 采样轨迹优化 refinement；长时域高动态 | [DynaRetarget 实体](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md) · [SBTO 方法](../../wiki/methods/dynaretarget-sbto-motion-retargeting.md) |
| ReActor | 仿真内双层 RL + 物理可行性闭环 | [ReActor 方法](../../wiki/methods/reactor-physics-aware-motion-retargeting.md) |
| OmniRetarget | Interaction Mesh + Laplacian；ICRA 2026 双最佳 | [OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) · [holosoma](../../wiki/entities/holosoma.md) |

## 对 wiki 的映射（汇总）

- Query 总览：[motion-retargeting-three-routes-landscape](../../wiki/queries/motion-retargeting-three-routes-landscape.md)
- 概念枢纽：[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[重定向管线](../../wiki/concepts/motion-retargeting-pipeline.md)
- **消歧：** AdaMorph（2601.07284）≠ [UMR 表面点云对应](./../../wiki/entities/paper-umr-unified-motion-retargeting.md)；GMR（YanjieZe）≠ Disney Generative Motion Rig 同名缩写

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 文内论文/项目映射到独立 wiki 节点（MoReFlow、AdaMorph 本次新建；其余复用已有页）
