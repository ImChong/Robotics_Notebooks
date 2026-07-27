# XPolicyLab（统一策略训推与评测适配层）

> 来源归档

- **标题：** XPolicyLab: A unified standard and infrastructure for robot policy development and deployment
- **类型：** repo / policy serving + adapter framework
- **组织：** XPolicyLab
- **代码：** <https://github.com/XPolicyLab/XPolicyLab>
- **文档/站：** <https://xpolicylab.github.io>（RoboDojo protocol 外链）
- **关联基准：** [RoboDojo](https://robodojo-benchmark.com/)、[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- **Stars：** ~117（2026-07-27）
- **License：** **Apache-2.0**
- **入库日期：** 2026-07-27
- **一句话说明：** 策略与评测环境之间的共享层：每个模型保留自有依赖/权重/训练配方于 `policy/<POLICY>/`，XPolicyLab 统一 **serving、观测/动作契约、eval 接线**；官方 RoboDojo / RoboTwin 上榜要求经本仓 PR 接入并可复现 checkpoint。

## 开源核查（2026-07-27）

- **已开源**：适配标准、`demo_policy` 参考实现、数据下载脚本（RoboDojo HDF5 / LeRobot / real）、debug/sim 评测路径。
- **`policy/` 目录（核查日）：** 约 **41** 个适配目录（含 `demo_policy`、ACT、DP 等基线），覆盖 A1、π₀/π₀.₅、GR00T N1.7、InternVLA、starVLA、Xiaomi_Robotics_0/1、GO1、RDT、OpenVLA-OFT 等；用户公告口径「**40+ 前沿模型复现**」与此一致（论文摘要写 integrate **30** policies — 以仓内当前目录为准并注明版本日）。

## 适配契约（README）

| 方法 | 合约 |
|------|------|
| `__init__(model_cfg)` | 加载配置、checkpoint、processor、`deploy.yml` 覆盖 |
| `update_obs` / `update_obs_batch` | 写入观测 |
| `get_action` / `get_action_batch` | 返回 action chunk（字典列表） |
| `reset()` | episode 间清状态 |

默认 policy-server 协议：`protocol: ws`（websocket）。

官方榜单 PR 规则要点：标准 adapter 布局；README 可复现 install/data/train/eval；本地至少 `EVAL_ENV_TYPE=debug`；上榜需附 checkpoint 下载脚本（HF / ModelScope 优先）。

## 对 wiki 的映射

- 实体页：[XPolicyLab](../../wiki/entities/xpolicylab.md)
- 评测基准：[RoboDojo](../../wiki/entities/robodojo.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
