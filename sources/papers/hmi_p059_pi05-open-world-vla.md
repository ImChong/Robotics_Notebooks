# π0.5: A Vision-Language-Action Model with Open-World Generalization（π0.5，HMI P059）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** π0.5: A Vision-Language-Action Model with Open-World Generalization
- **短名：** π0.5
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P059
- **年份：** 2025
- **原文：** https://arxiv.org/abs/2504.16054
- **代码：** https://github.com/Physical-Intelligence/openpi
- **项目页：** https://www.physicalintelligence.company/blog/pi05
- **入库日期：** 2026-07-31
- **一句话说明：** 预训练用 FAST 离散动作吃异构数据，后训练再为目标本体接入连续 flow 专家；推理时先出语义子任务再高频生成动作块。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P059](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P059.md)

## 开源状态（步骤 2.5）

- **结论：** 部分开源（openpi；完整数据与训练管线未全部公开）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

π0.5不是简单把π0换一批数据继续训练。它面向家庭移动操作的长时任务，把统一模型拆成两个训练阶段和两种推理节奏：预训练用离散表示吃下异构数据，后训练再为目标本体加入连续动作专家。执行时同一模型先产生语义子任务，然后以它为条件生成低层动作。

**对 wiki 的映射：** [`wiki/entities/paper-pi05-open-world-vla.md`](../../wiki/entities/paper-pi05-open-world-vla.md)

### 摘录 2

Flow matching很适合精细连续控制，但不方便与文本、网页问答和只有语义标注的数据用同一个自回归目标混合。π0.5先用FAST将连续动作压缩成离散token，与多本体轨迹、高层语义预测和web数据共同预训练。这个阶段的目标不是达到最精细的底层控制，而是建立场景、任务、本体和动作语义之间的广泛对应。

**对 wiki 的映射：** [`wiki/entities/paper-pi05-open-world-vla.md`](../../wiki/entities/paper-pi05-open-world-vla.md)

### 摘录 3

FAST token保留一段动作的时序结构，使动作数据可以和“下一步子任务是什么”这类文本监督共用自回归训练接口；本体、相机和语言仍作为条件。后训练时连续flow expert重新接管精细动作输出，因此离散预训练表示负责迁移语义，连续头负责目标机器人控制。两阶段使用的动作表示不同，必须有明确的本体适配和对齐，不能把FAST token直接当成电机命令。

**对 wiki 的映射：** [`wiki/entities/paper-pi05-open-world-vla.md`](../../wiki/entities/paper-pi05-open-world-vla.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-pi05-open-world-vla.md`](../../wiki/entities/paper-pi05-open-world-vla.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
