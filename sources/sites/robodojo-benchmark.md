# RoboDojo 官网与长期公益评测入口

> 来源归档

- **标题：** RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies
- **类型：** site（官网 + 文档 Wiki + Leaderboard + Eval 入口）
- **URL：** <https://robodojo-benchmark.com/>
- **文档：** <https://robodojo-benchmark.com/doc/>
- **榜单：** <https://robodojo-benchmark.com/leaderboard>
- **上榜入口与规则：** <https://robodojo-benchmark.com/eval>
- **评测协议（Integrity / Anti-Gaming）：** <https://robodojo-benchmark.com/leaderboard/protocol>
- **论文：** <https://arxiv.org/abs/2607.04434>
- **代码：** <https://github.com/RoboDojo-Benchmark/RoboDojo>
- **策略基建：** <https://github.com/XPolicyLab/XPolicyLab>（文档亦链 <https://xpolicylab.github.io>）
- **运营主体：** AI MMLab Club（非营利基金会）；全球学术机构联合公益运行，**不接受商业公司治理 / 资助 / 算力赞助**（官网 README / protocol 声明）
- **入库日期：** 2026-07-27
- **一句话说明：** 通用机器人操纵 **仿真+真机统一评测** 官网：开放长期公益榜单与线上评测入口；官方 verified 上榜要求经官方评测管线、隐藏布局校验，并在分数公开前经 **XPolicyLab** 开源训推代码、评测 checkpoint、配置与复现说明，同时公布评测视频。

## 开源核查（步骤 2.5，2026-07-27）

| 资源 | 状态 | 说明 |
|------|------|------|
| 仿真评测栈 | **已开源** | [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)：任务 / 资产校验 / Isaac Sim client / `summarize`；本 release 为 **eval-only** |
| 策略适配与复现 | **已开源** | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)：统一 policy server 接口；截至核查日 `policy/` 下约 **40+** 前沿模型适配目录 |
| 真机评测 | **部分开放** | RoboDojo-RealEval：远程云评测 + 标准化硬件 / 场景复位 / 协议；官方榜单走云端管线与防作弊校验 |
| 权重 | **按上榜规则强制开源** | Private 迭代可不公开；**verified leaderboard 公布阶段**须释放 evaluated checkpoint |

## 长期公益评测 / 上榜规则（官网 Eval + Protocol，策展）

治理与公正性要点（与用户公告一致，细节以 protocol 页为准）：

1. **治理：** AI MMLab Club 维护榜单；全球学术伙伴共治；无商业资助 / 赞助。
2. **Private vs Verified：** 可用远程 policy server 做私有评测迭代，**不必**先开源代码/权重；无完整评测产物的结果单独标注，**不算 verified 条目**。
3. **官方 verified 上榜要件：**
   - 经 **官方线上评测系统**（提交可部署包或连接远程 policy server）
   - 仿真：**三随机种子** mean ± std；真机：覆盖 **ARX X5 / Piper / Piper X** 三本体
   - 通过 **hidden-layout verification**（公开布局为主榜；隐藏布局作一致性辅助，防过拟合 / 刷榜）
   - 在分数对外公布前，经 **XPolicyLab** 释放：训推与部署代码、evaluated checkpoint、配置、加载推理与统一接口下的评测说明
   - **公布评测视频**供社区检查（每模型×任务按 seed 抽样视频；站内文案示例：每 seed 2 条、共 6 条量级）
4. **入口：** [Eval](https://robodojo-benchmark.com/eval) · [Leaderboard](https://robodojo-benchmark.com/leaderboard) · [Protocol](https://robodojo-benchmark.com/leaderboard/protocol)

## 页面结构（维护索引）

| 路径 | 内容要点 |
|------|----------|
| `/` | 统一 sim-and-real 基准叙事、五能力维、生态组件 |
| `/doc/` | Starlight 文档：安装、XPolicyLab、Quick Eval、42 sim / 18 real 任务 |
| `/leaderboard` | 在线榜单 |
| `/leaderboard/protocol` | 评测完整性与反刷榜协议 |
| `/eval` | 上榜入口与规则说明 |
| `/community` | 社区入口 |

## 对 wiki 的映射

- 主实体：[RoboDojo](../../wiki/entities/robodojo.md)
- 策略基建：[XPolicyLab](../../wiki/entities/xpolicylab.md)
- 论文摘录：[robodojo_arxiv_2607_04434.md](../papers/robodojo_arxiv_2607_04434.md)
- 公告归档：[robodojo_open_longterm_eval_2026-07.md](../blogs/robodojo_open_longterm_eval_2026-07.md)
- 交叉：[具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)、[VLA](../../wiki/methods/vla.md)、[Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md)
