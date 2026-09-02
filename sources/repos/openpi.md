# openpi（Physical Intelligence）

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** openpi
- **类型：** repo
- **技术路线分组：** 世界模型、VLA与Agent
- **链接：** https://github.com/Physical-Intelligence/openpi
- **入库日期：** 2026-07-30
- **一句话说明：** 仓库同时维护流匹配式π0、快速自回归π0-FAST和π0.5，并提供检查点、数据配置、微调与推理服务。接入新机器人时最关键的工作是动作归一化、数据字段映射和推理频率对齐。
- **开源状态（据主表）：** 已开源（以官方仓库 README 为准）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 方法页 [`wiki/methods/π0-policy.md`](../../wiki/methods/π0-policy.md)；论文详情 [`wiki/entities/paper-pi0.md`](../../wiki/entities/paper-pi0.md)

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [π0 论文实体](../../wiki/entities/paper-pi0.md) — arXiv:2410.24164 canonical 节点
- [π0-policy](../../wiki/methods/π0-policy.md) — HMI 开源主表入口 Physical-Intelligence/openpi（π0 / π0-FAST / π0.5 官方代码与权重）
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
- [χ₀ / kai0](../../wiki/entities/paper-kai0.md) — 基于 openpi 的协同叠衣后训练与部署对齐（OpenDriveLab/kai0）
