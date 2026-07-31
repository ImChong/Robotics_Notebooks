# Synthesis of Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives（Whole-Body Behaviors，HMI P002）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Synthesis of Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives
- **短名：** Whole-Body Behaviors
- **类型：** paper / hmi-papers / 工程与实机部署
- **HMI ID：** P002
- **年份：** 2005
- **原文：** https://doi.org/10.1142/S0219843605000594
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 把全身行为拆成有严格优先级的行为基元：高优先级先占用自由度，低优先级只在动态一致零空间内工作。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P002](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P002.md)

## 开源状态（步骤 2.5）

- **结论：** 不适用（经典论文）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

让人形机器人双手拿箱子时，手要到位，质心不能跑出支撑区域，躯干还要保持合适姿态。最直接的做法是给所有误差加权求和，但权重稍有变化，平衡任务就可能被手部任务牺牲。这篇论文的核心思想是把行为拆成有严格优先级的primitive：高优先级任务先占用它需要的自由度，低优先级任务只能在不破坏前者的剩余空间里工作。

**对 wiki 的映射：** [`wiki/entities/paper-whole-body-behaviors-primitives.md`](../../wiki/entities/paper-whole-body-behaviors-primitives.md)

### 摘录 2

任务层在操作空间中描述手、质心或躯干等目标，利用任务空间动力学计算相应控制力；姿态层处理冗余关节，希望机器人在完成任务的同时维持自然或可控的全身构型。每增加一个低优先级任务，都要先投影到前面所有高优先级任务的动态一致零空间。于是“拿箱子”不是手、质心和躯干三个控制器简单叠加，而是一个有明确支配关系的递归结构。

**对 wiki 的映射：** [`wiki/entities/paper-whole-body-behaviors-primitives.md`](../../wiki/entities/paper-whole-body-behaviors-primitives.md)

### 摘录 3

这种做法的价值在冲突时最明显：如果双手目标和保持平衡不能同时满足，系统应先保住平衡，再在剩余能力内尽量靠近手部目标。严格层级比加权和更容易表达这种安全语义，但也意味着优先级选错会造成低层任务长期没有自由度，任务切换还可能带来不连续。

**对 wiki 的映射：** [`wiki/entities/paper-whole-body-behaviors-primitives.md`](../../wiki/entities/paper-whole-body-behaviors-primitives.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-whole-body-behaviors-primitives.md`](../../wiki/entities/paper-whole-body-behaviors-primitives.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
