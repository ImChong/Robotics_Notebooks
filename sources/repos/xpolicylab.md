# XPolicyLab（统一策略训推与评测适配层）

> 来源归档

- **标题：** XPolicyLab: A unified standard and infrastructure for robot policy development and deployment
- **类型：** repo / policy serving + adapter framework
- **组织：** XPolicyLab Community（MMLab@HKU & THU）
- **代码：** <https://github.com/XPolicyLab/XPolicyLab>
- **文档/站：** <https://xpolicylab.github.io/> — [`sources/sites/xpolicylab-github-io.md`](../sites/xpolicylab-github-io.md)
- **论文：** <https://arxiv.org/abs/2608.09892> — [`sources/papers/xpolicylab_arxiv_2608_09892.md`](../papers/xpolicylab_arxiv_2608_09892.md)
- **关联基准：** [RoboDojo](https://robodojo-benchmark.com/)、[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- **Stars：** ~145（2026-08-12）
- **License：** **Apache-2.0**
- **入库日期：** 2026-07-27；**刷新：** 2026-08-12（对齐论文 42 策略与技术报告）
- **一句话说明：** 策略与评测环境之间的共享层：每个模型保留自有依赖/权重/训练配方于 `policy/<POLICY>/`，XPolicyLab 统一 **serving、观测/动作契约、eval 接线**；官方 RoboDojo / RoboTwin 上榜要求经本仓 PR 接入并可复现 checkpoint。

## 开源核查（2026-08-12）

- **已开源**：适配标准、`demo_policy` 参考实现、client/server（`client_server/`）、数据下载脚本、debug/sim 评测路径、Cursor/Claude agent skills（`.agents` / `.cursor` / `.claude`）。
- **`policy/` 目录：** 论文 Table I / 项目页口径 **42** 策略（2026-08-08）；引用「多少模型」时以仓内当日目录为准并注明核查日。

## 适配契约（README / 论文）

| 方法 | 合约 |
|------|------|
| `__init__(model_cfg)` | 加载配置、checkpoint、processor、`deploy.yml` 覆盖 |
| `update_obs` / `update_obs_batch` | 写入观测 |
| `get_action` / `get_action_batch` | 返回 action chunk（字典列表） |
| `reset()` | episode 间清状态 |

默认 policy-server 协议：`protocol: ws`（WebSocket + MessagePack）。

官方榜单 PR 规则要点：标准 adapter 布局；README 可复现 install/data/train/eval；本地至少 `EVAL_ENV_TYPE=debug`；上榜需附 checkpoint 下载脚本（HF / ModelScope 优先）。

## 关键复现入口

```bash
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab && pip install -e .
bash scripts/create_policy.sh <POLICY_NAME>
cd policy/<POLICY_NAME>
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 <policy_env> base
```

## 对 wiki 的映射

- 实体页：[XPolicyLab](../../wiki/entities/xpolicylab.md)
- 评测基准：[RoboDojo](../../wiki/entities/robodojo.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
