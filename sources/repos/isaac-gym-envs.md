# IsaacGymEnvs

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** IsaacGymEnvs
- **类型：** repo
- **技术路线分组：** 工程与实机部署
- **链接：** https://github.com/isaac-sim/IsaacGymEnvs
- **入库日期：** 2026-07-30
- **一句话说明：** 汇集Isaac Gym时代的GPU向量化强化学习任务、训练配置和多算法接入方式，便于复现早期官方基线；其价值主要在历史任务实现与性能对照，新任务开发应优先评估Isaac Lab迁移路径。
- **开源状态（据主表）：** 已开源（以官方仓库 README 为准）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-gym.md`](../../wiki/entities/isaac-gym.md)（不另建重复实体；合并入已有节点）

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [isaac-gym](../../wiki/entities/isaac-gym.md) — HMI 开源主表入口 isaac-sim/IsaacGymEnvs（Isaac Gym 时代官方任务与训练配置仓）
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)

## 官方原文复核（2026-11-17）

- **版本：** `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`。
- **核对范围：** 官方 README（安装、任务、参数、训练、checkpoint、随机化、多卡与视频）；对照 `isaacgymenvs/train.py` 环境注册和 Runner 入口、`tasks/base/vec_task.py` 的 step 实现；未运行 GPU 训练、推理或真机实验。
- **通读：** 已完成

官方任务仓可运行实现公开；Isaac Gym Preview 二进制须另行下载，不能把任务仓开源理解为 PhysX/Gym 全部源码开放。`VecTask.step` 只有 CPU pipeline 分支显式调用 `fetch_results`。
