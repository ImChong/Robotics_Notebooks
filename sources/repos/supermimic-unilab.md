# Uni-SuperMimic

> 来源归档

- **标题：** Uni-SuperMimic
- **类型：** repo（独立复现）
- **链接：** <https://github.com/Motphys/Uni-SuperMimic>
- **论文：** <https://arxiv.org/abs/2610.28480>（SuperMimic）
- **参考实现：** [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
- **机构：** Motphys
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-22
- **再核日期：** 2026-10-22
- **一句话说明：** 基于 UniLab 对 SuperMimic 做独立复现，面向 Unitree G1 的 LAFAN1 全身动作跟踪，提供 mjwarp、MuJoCo 与 Motrix 后端以及训练、评测和渲染入口。
- **沉淀到 wiki：** [SuperMimic](../../wiki/entities/paper-supermimic.md)

---

## 开源状态（2026-10-22）

| 项 | 状态 |
|---|---|
| 训练与评测代码 | **已开源**；README 提供 unisupermimic-train 训练、续训与评测命令 |
| 仿真后端 | mjwarp、MuJoCo、Motrix；README 说明 ROCm 主机需使用 MuJoCo host-bridge 路径 |
| 资产 | 由工具从 Hugging Face 的 LAFAN1 G1 数据集下载；运动 CSV、URDF/mesh 等不随仓库分发 |
| 关系 | 独立下游复现；README 明确引用 supermimic_vibed 并依赖 PyPI unilab，不是论文作者的官方仓库 |

## 实现线索

代码按 UniLab 包组织：src/unisupermimic/rl/algos/supermimic/ 包含世界模型、策略、critic、回放缓存和 learner；G1 motion-tracking task 下提供环境、manager terms、动作库、评测、渲染和注册入口。README 的资产校验、严格 Sim2Sim 检查和从动作首帧开始的整段评测，有助于区分训练成功、动作库覆盖和模型 I/O 一致性。

## 对 wiki 的映射

- [SuperMimic：论文与研究实体](../../wiki/entities/paper-supermimic.md) — 与官方论文共用实体节点，注明这是独立实现。
- [SuperMimic 论文来源](../papers/supermimic_arxiv_2610_28480.md) — 原始方法与评测主张。
- [官方项目页](../sites/supermimic-project.md) — 官方项目页未列作者维护代码，勿将本仓库误标为官方实现。
