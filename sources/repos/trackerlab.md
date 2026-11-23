# TrackerLab

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** TrackerLab
- **类型：** repo
- **技术路线分组：** 动作跟踪与全身控制
- **链接：** https://github.com/Renforce-Dynamics/trackerLab
- **入库日期：** 2026-07-30
- **最后核对：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全仓核查 `Renforce-Dynamics/trackerLab`（`master` 分支、README、`docs/data_flow.md`、`poselib/scripts/amass/retarget_all.py`、`trackerLab/commands/manager/`、`trackerTask/trackerLab/`；核实官方运行入口实为 `scripts/rls/rsl_rl/base/train.py` 与 `play.py`，而 README 中的 `scripts/rsl_rl/base/` 路径已失效；含 ONNX/JIT 导出与 `deploylib` 部署链路）
- **一句话说明：** 多种本体、动作数据和跟踪配置共享实验接口，可把参考表示、奖励与机器人模型的影响分开比较。已有Tracker迁移到新本体后若突然退化，这个统一基线有助于先排除环境差异。
- **开源状态（2026-11-21 复核）：** 已开源（覆盖 H1 / G1 / K1 / Pi / R2 等本体、`beyondMimic` 与 `locomotion_rl_lab` 配置）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/trackerlab.md`](../../wiki/entities/trackerlab.md)

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [TrackerLab](../../wiki/entities/trackerlab.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
