# AnyBody: Free-Form Whole-Body Humanoid Control from Arbitrary Keypoint Guidance（AnyBody，HMI P039）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** AnyBody: Free-Form Whole-Body Humanoid Control from Arbitrary Keypoint Guidance
- **短名：** AnyBody
- **类型：** paper / hmi-papers / 动作跟踪与全身控制
- **HMI ID：** P039
- **年份：** 2026
- **原文：** https://arxiv.org/abs/2606.29209
- **代码：** https://github.com/hazel-hammer/Anybody
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 把物理技能压入潜在解码器，再用关键点条件补全器从任意稀疏关键点子集推断可行全身意图并执行。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P039](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P039.md)

## 开源状态（步骤 2.5）

- **结论：** 已开源（hazel-hammer/Anybody）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

已有稀疏控制器通常预先规定输入是头和双手，或限定若干command mask。AnyBody希望部署时才决定给哪些身体关键点：这次只给双手，下次增加脚和头，甚至不同时间使用不同子集。方法先建立统一latent motion space，再训练一个masked Transformer把任意关键点集合投到这个空间。

**对 wiki 的映射：** [`wiki/entities/paper-anybody-keypoint-humanoid-control.md`](../../wiki/entities/paper-anybody-keypoint-humanoid-control.md)

### 摘录 2

特权teacher在大规模无结构动作库上做全身tracking。在线蒸馏时，deterministic encoder把完整运动目标编码到单位球面latent，decoder读取latent与当前本体状态并拟合teacher动作。球面约束固定latent尺度，降低不同动作表示的漂移；更重要的是decoder始终看到当前状态，因此包含接触与平衡反馈，不是静态pose解码器。

**对 wiki 的映射：** [`wiki/entities/paper-anybody-keypoint-humanoid-control.md`](../../wiki/entities/paper-anybody-keypoint-humanoid-control.md)

### 摘录 3

第二阶段冻结latent空间和decoder。每个关键点目标被当作token，缺失点通过masked self-attention自然忽略；Transformer encoder根据任意子集预测teacher latent。训练中随机抽取子集，才使部署时的自由组合成为分布内问题。最终同一decoder把不同输入组合转成全身动作。

**对 wiki 的映射：** [`wiki/entities/paper-anybody-keypoint-humanoid-control.md`](../../wiki/entities/paper-anybody-keypoint-humanoid-control.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-anybody-keypoint-humanoid-control.md`](../../wiki/entities/paper-anybody-keypoint-humanoid-control.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
