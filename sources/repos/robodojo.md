# RoboDojo（官方评测仓库）

> 来源归档

- **标题：** RoboDojo Official Repo
- **类型：** repo / benchmark toolkit（eval-only）
- **组织：** RoboDojo-Benchmark；运营 AI MMLab Club + 学术联盟
- **代码：** <https://github.com/RoboDojo-Benchmark/RoboDojo>
- **主页：** <https://robodojo-benchmark.com/>
- **论文：** <https://arxiv.org/abs/2607.04434>
- **Stars：** ~304（2026-07-27）
- **License：** 根目录 `LICENSE` 为 **MIT**（版权声明 Yue Chen, 2025）；README 徽章/文案另写 Non-Commercial Research — **以 LICENSE 文件为准**，引用时注明 README 文案不一致
- **技术栈徽章：** Python 3.11 · Isaac Sim 5.1 · Isaac Lab 2.3
- **入库日期：** 2026-07-27
- **一句话说明：** RoboDojo **评测侧**开源仓：Isaac Sim 异构并行仿真客户端、42/18 任务与资产配置、结果汇总；**策略集成与训推**归 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)。

## 仓库结构（README）

```text
env/                   simulator backbone and managers
env_cfg/               simulator, scene, robot, and camera configs
task/RoboDojo/         task logic and task YAML configs
scripts/robodojo.sh    public RoboDojo-side eval entry
scripts/eval_policy.sh simulator client launched by XPolicyLab eval.sh
XPolicyLab/            policy server and policy integrations（子模块/对接）
Assets/                downloaded robot, object, material, and layout assets
```

策略侧约定每个 policy 提供：

```text
XPolicyLab/policy/<POLICY_NAME>/eval.sh
XPolicyLab/policy/<POLICY_NAME>/deploy.yml
```

## 能力与任务规模（README / 论文）

| 维度 | 规模 |
|------|------|
| 仿真任务 | **42**（五维：Generalization / Memory / Precision / Long-Horizon / Open） |
| 真机任务 | **18**（本体：Piper X、Piper、ARX X5） |
| 并行仿真 | 异构并行（不同任务/场景/进程并发于 Isaac Sim） |
| 资产 | 刚体 / 铰接 / 可变形物体，配置驱动场景 |

## 对 wiki 的映射

- 实体页：[RoboDojo](../../wiki/entities/robodojo.md) — 统一 sim-and-real 评测、上榜规则、与 XPolicyLab 分工。
- 姊妹仓：[XPolicyLab](./xpolicylab.md)
- 项目页：[robodojo-benchmark.md](../sites/robodojo-benchmark.md)
