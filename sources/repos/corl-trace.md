# corl-trace（Jeong-zju/corl-trace）— 原始资料归档

- **来源：** <https://github.com/Jeong-zju/corl-trace>
- **类型：** repo
- **论文：** [TRACE（arXiv:2606.14551）](https://arxiv.org/abs/2606.14551)
- **项目页：** <https://jeong-zju.github.io/trace/>
- **机构：** 芝诺机器人（Zeno AI）；浙江大学（ZJU）；浙江工业大学（ZJUT）；悉尼大学（USYD）
- **归档日期：** 2026-09-15
- **默认分支：** `master`

## 一句话说明

**TRACE** 官方实现：固定槽因果记忆 + 路径签名（`signatory` 依赖）在线更新；以 **Streaming ACT** 等轻量 adapter 挂到 ACT / Diffusion 骨干；含 Meta-World / RoboCasa / **zeno-ai** 真机数据配置与 ROS1 闭环部署。

## 开源核查（步骤 2.5）

| 项 | 结论（截至 2026-09-15） |
|----|-------------------------|
| **代码** | **已开源** — 训练 `bash/train_policy.sh`、采集 `scripts/collect_imitation_dataset.py`、评测 `bash/eval_policy.sh` |
| **策略包** | `policy/lerobot_policy_streaming_act`（Streaming ACT + signature cache） |
| **部署** | `deploy/ros1_adapter/ros1_adapter_node.py` + `deploy/configs/deploy_zeno_act.yaml` 等 |
| **权重 / 私有数据** | 仓库 **不含** 论文全量真机 checkpoint；`bash/defaults/zeno-ai/` 为配置模板，数据需自行准备 |
| **环境** | `environment.yml`；`depends/signatory` 子模块 |

## 目录要点

| 路径 | 作用 |
|------|------|
| `scripts/collect_imitation_dataset.py` | 示教采集（`--path-signature-depth 3`） |
| `bash/train_policy.sh` | ACT / diffusion / streaming_act 训练入口 |
| `policy/lerobot_policy_streaming_act/` | TRACE Streaming ACT 模型与 signature 缓存 |
| `deploy/` | 单进程 ROS1：订阅图像+双臂状态+里程计 → `select_action()` → 发布 Twist/JointState |
| `benchmarks/` | RoboCasa 等评测辅助 |

## 对 wiki 的映射

- [paper-trace-causal-memory](../../wiki/entities/paper-trace-causal-memory.md)
- [trace.md](../sites/trace.md)
- [trace_causal_memory_arxiv_2606_14551](../papers/trace_causal_memory_arxiv_2606_14551.md)
