# ACT（Action Chunking Transformer）

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** ACT
- **类型：** repo
- **技术路线分组：** 世界模型、VLA与Agent
- **链接：** https://github.com/tonyzhaozh/act
- **入库日期：** 2026-07-30
- **一句话说明：** 以条件变分自编码器和Transformer一次预测动作块，再用时间集成平滑连续控制，减少长任务中的逐步误差累积；低成本双臂数据采集与真实部署代码使其成为模仿学习常用基线。
- **开源状态（据主表）：** 已开源（以官方仓库 README 为准）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 方法页 [`wiki/methods/action-chunking.md`](../../wiki/methods/action-chunking.md)；论文详情 [`wiki/entities/paper-act.md`](../../wiki/entities/paper-act.md)

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [ACT 论文实体](../../wiki/entities/paper-act.md) — arXiv:2304.13705 canonical 节点
- [action-chunking](../../wiki/methods/action-chunking.md) — HMI 开源主表入口 tonyzhaozh/act；官方双臂 ACT 实现仓
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
