# XPolicyLab（arXiv:2608.09892）

> 来源归档（ingest）

- **标题：** XPolicyLab: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment
- **缩写：** **XPolicyLab**
- **类型：** paper / infrastructure / policy-serving / evaluation / open-ecosystem
- **arXiv：** <https://arxiv.org/abs/2608.09892>
- **HTML：** <https://arxiv.org/html/2608.09892>
- **PDF：** <https://arxiv.org/pdf/2608.09892>
- **项目页：** <https://xpolicylab.github.io/> — 归档见 [`sources/sites/xpolicylab-github-io.md`](../sites/xpolicylab-github-io.md)
- **代码：** <https://github.com/XPolicyLab/XPolicyLab>（Apache-2.0）— 归档见 [`sources/repos/xpolicylab.md`](../repos/xpolicylab.md)
- **作者 / 主导机构：** XPolicyLab Community；MMLab@HKU & THU；项目牵头 Tianxing Chen；通讯 Wenbo Ding、Ping Luo
- **机构：** 香港大学（HKU）MMLab；清华大学（Tsinghua）
- **入库日期：** 2026-08-12
- **一句话说明：** 把「N 策略 × M 评测环境」的 \(O(NM)\) 集成降为 \(O(N{+}M)\)：统一观测/动作/轨迹 schema + 最小 adapter 契约 + 依赖隔离的 client/server；截至 2026-08 集成 **42** 策略，同一 adapter 服务 RoboTwin、RoboDojo-sim 与 RoboDojo-real。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2608.09892>
- **核心贡献：** 策略侧依赖、数据表示与运行时接口碎片化，导致跨基准/仿真/真机重复接线且易 silently 改相机命名、通道序、夹爪缩放。XPolicyLab 提供统一契约与依赖隔离 serving，使策略保留原生栈、环境侧只写一份 client。
- **对 wiki 的映射：**
  - [XPolicyLab 论文实体](../../wiki/entities/paper-xpolicylab.md)
  - [XPolicyLab 工具实体](../../wiki/entities/xpolicylab.md)
  - [RoboDojo](../../wiki/entities/robodojo.md)

### 2) Adapter 契约与依赖隔离 serving（§III）

- **链接：** §III-B / III-C
- **核心贡献：**
  - 四操作：`__init__` / `update_obs`(+batch) / `get_action`(+batch) / `reset`。
  - 进程隔离：policy server 与 env client 经 WebSocket + MessagePack（数组扩展）；消息集含 HELLO、PREPARE_CASE、RESET、INFER、CALL、TRIAL_END、HEARTBEAT、CLOSE。
  - 可靠性：请求 ID + 响应缓存防重试双推理；server instance ID 变化视为致命（状态丢失）。
  - 观测 schema：\(\mathbf{o}_t=\{\mathbf{v}_t,\mathbf{q}_t,\mathbf{p}_t,\ell,\mathbf{m}_t\}\)；Cartesian 约定 \([x,y,z,q_w,q_x,q_y,q_z]\)；图像解码在 serving 层统一。
- **对 wiki 的映射：**
  - [XPolicyLab 论文实体](../../wiki/entities/paper-xpolicylab.md) — 源码运行时序图
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)

### 3) 生态规模与集成代价（§IV / 项目页）

- **链接：** Experiments / <https://xpolicylab.github.io/>
- **核心贡献：**
  - **42** 策略覆盖 VLA / WAM / 扩散 / 记忆增强 / 经典 IL（Table I，截至 2026-08-08）。
  - 模型侧代码量级差一个数量级，环境侧闭环仍落在固定参考的几行内。
  - 受控研究：π₀.₅ → RoboDojo 集成 **>5 h → 2 h**；打包 agent skills 再降至约 **30 min**。
  - 同一 adapter：RoboTwin、RoboDojo-sim、RoboDojo-real / RealEval。
- **对 wiki 的映射：**
  - [RoboDojo](../../wiki/entities/robodojo.md)
  - [具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)

### 4) 开源边界（步骤 2.5）

- **链接：** 项目页 / GitHub
- **核心贡献：** **已开源** — 框架、adapter 标准、`demo_policy`、评测脚本、agent skills；各 `policy/<NAME>/` 自管依赖与 checkpoint 下载脚本。Stars ≈145（2026-08-12）。
- **对 wiki 的映射：**
  - [仓库归档](../repos/xpolicylab.md)
  - [项目页归档](../sites/xpolicylab-github-io.md)

## 对 wiki 的映射（汇总）

- 论文实体：[`wiki/entities/paper-xpolicylab.md`](../../wiki/entities/paper-xpolicylab.md)
- 工具实体刷新：[`wiki/entities/xpolicylab.md`](../../wiki/entities/xpolicylab.md)
- 交叉：[RoboDojo](../../wiki/entities/robodojo.md)、[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)、[具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)、[VLA](../../wiki/methods/vla.md)

## BibTeX（项目页）

```bibtex
@misc{community2026xpolicylabunifiedstandardopen,
  title={XPolicyLab: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment},
  author={XPolicyLab Community and Tianxing Chen and Yue Chen and Tian Nian and Zijian Cai and Guangyu Chen and Wenwei Lin and Qiwei Liang and others},
  year={2026},
  eprint={2608.09892},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2608.09892}
}
```
