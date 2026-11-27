# You Don't Need To Stay in The Loop: An Agentic Robotics Loop for Robot-Policy Improvement

- **作者：** Hang Yu (`hangyu8123@gmail.com`)
- **机构：** Independent Researcher, United States
- **版本 / 载体：** arXiv:2608.07555v2 [cs.RO]（2026-08-02 提交 v1，2026-11-08 修订 v2，标注 "Agentic Robotics Preview Version"）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2608.07555) / [arXiv Abs](https://arxiv.org/abs/2608.07555)
- **项目页 / 代码：** 未确认开源代码（论文标注预览仓库地址 [GitHub](https://github.com/HangYu8123/AgenticRobotics/tree/af89f02d8f886946b1870b75c166e514b75d7cc5)，经 2026-11-27 核查该仓库返回 404 / 未公开可访问；正文 §6 提及 `tools/`、`agentic_robot/tools.py`、`agentic_robot/invocation.py`、`agentic_robot/mcp.py` 及 `tech_report/experiments`，但当前无可获取的公开源码实现）
- **核对范围：** arXiv:2608.07555v2 全文（Abstract、§1–§7、Figure 1、Table 1–3 及 References）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心定位与边界声明（§1, §3.1, §7）：** 论文研究的是「AI 智能体接管机器人策略改进的外层研究循环（outer research loop）」，而非机器人在物理世界中的具身行动。作者明确将标题限定为**操作性主张（operational claim）**而非**选型优势主张（selection claim）**：人工可以离开循环，是因为晋升受证据门控、状态可从崩溃中恢复、能力质量由记录派生且随产出物过期，而不是因为循环选出的检查点比人类更优（在所测单条谱系上并未优于「直接取最后一个检查点」）。
- **控制面与后端分离架构（§3.1–§3.4, Fig. 1）：**
  - **四条设计准则（§3.2）：** （1）万物工具化（所有策略、训练器、评估器、遥操采集等均封装在描述符后，经由单一标准化 MCP `input -> tool -> output` 界面调用，6 个种子描述符为 policy、trainer、evaluator、recorder、dataset editor、calibrator）；（2）只信实测不信描述（能力质量分为 `unvalidated`、`effective`、`ineffective`、`stale`、`deprecated` 五级，仅由控制器解析结果写入，按 Wilson 下界排序，产物引用变动即自动转为 `stale`）；（3）技能是工具之上的可调工作流（Agent Skills Markdown 格式，29 个循环技能 + 1 个作者工具，通过 `agent -> skill -> tool` 解耦意图、流程与后端执行）；（4）剥离物理细节（控制器不直接感知机器人本体，硬件与仿真均置于工具边界之下）。
  - **研究回合事务与 11 条不变量（§3.4）：** 不可变目标（metric / direction / threshold / eval command / run dir / optional acceptance holdout），每回合严格按 $\mathrm{Bind} \rightarrow \mathrm{Analyze} \rightarrow \mathrm{Act} \rightarrow \mathrm{Measure} \rightarrow \mathrm{Score} \rightarrow \mathrm{Commit}$ 推进；单一在途记录通过原子替换与 sidecar 锁以提交键分阶段推进，崩溃恢复补齐首个缺失阶段而不重放已提交副作用；通过 generation ID 防止延迟 worker/timer 唤醒新代控制器。
- **证据门控与安全机制（§3.5–§3.6, §5）：**
  - **Agresti–Caffo 双样本门控（式 1）：** $\tilde{p}_i = \frac{k_i+1}{n_i+2}$，$w = 1.96\sqrt{\sum_{i=1}^2 \frac{\tilde{p}_i(1-\tilde{p}_i)}{n_i+2}}$；当 $\Delta = m_{\text{new}} - m_{\text{best}} \ge 1.5w$ 直接晋升，$0 < \Delta < 1.5w$ 触发配对 McNemar 或配对 bootstrap 确认，无提升则累计停滞计数以强制切换策略类别但不终止运行。
  - **任意时刻有效序贯检验与防篡改/幂等/围栏（§5）：** 针对可选停止（optional stopping），评估了 betting $e$-process（要求下注比例 $\lambda < 2$ 以维持非负鞅）与 mixture-SPRT；给出带 `(run_id, round_id, effect_type, generation)` 的后端幂等 API 契约、非 LLM 解析器 + HMAC 摘要绑定、以及基于栅栏令牌（fenced token）的聚合器。
- **案例研究与实验数据（§4–§5, Table 1–3）：**
  - **案例 1（§4.1 Task-5 VLA 筛选）：** 在 LIBERO-10 单任务（放书入盒，由 $n=10$ 探针选出基线最高任务）筛选 $\pi_{0.5}$、MolmoAct2、GR00T N1.7、X-VLA；10 个保留配置（$n=30$，固定种子）中 7 个达 $100\%$、全部达 $90\%$ 门槛，最大动作块 + 减少 flow-matching 步数将单回合耗时从 `29.0` 降至 `12.8 s/episode`；Task 2 扩展中 GR00T 与 chunk-50 $\pi_{0.5}$ 均测得 `83.3%`。
  - **案例 2（§4.2 SmolVLA 改进循环）：** 目标为 LIBERO-10 平均成功率 $\ge 70\%$，从 55% 检查点热启动（每轮 100 episodes），提交 0–26 轮（5 个日历日 / 约 3 天 GPU 活跃时间），第 27 轮未完成时由人工中止。第 2 轮线性任务向量融合（task-vector soup）测得 `63%` 成为记录冠军，后续数据重平衡、TIES 合并、自模仿等仅 `47–55%`。对该冠军汇总 `1,200` episodes 复测实际为 `56.5%`（12 次读数 SD `3.6` 点），证明历史 `+8` 点（McNemar $p \approx 0.25$）属于选择偏差而非真实能力跃升。
  - **运行特性实测（§5）：**
    - Gate validity（Table 2，$2\times 10^5$ 次 Monte-Carlo，null $p=0.565, n=100$/look，最多 1,200 episodes）：固定 Agresti–Caffo $1.5\times$ I 类错误 `0.002`；固定 min-delta $1.5\times$/12 pt I 类错误 `0.049`（+8 点功效 `0.31`）；McNemar 偷看膨胀至 `0.161`；$e$-process I 类错误 `0.003`（功效 `0.61`，平均样本 `920`）；Mixture-SPRT I 类错误 `0.007`（功效 `0.86`，平均样本 `700`）。配对初始状态带来共享潜在效应时上述序贯边界成立（Type-I `0.0027–0.0032`），若潜在效应为 arm-specific 则各类逐回合检验全部膨胀（`0.118–0.505`）。
    - 崩溃恢复：`14,000` 次进程杀死注入及 `78` 字节部分写入偏移均实现 `0` 丢失 / `0` 重复，恢复耗时 `0.15–0.25 ms`（仅覆盖账本层；实际 LeRobot 二进制未实现后端幂等键）。
    - 决策质量反例（Table 3，从头训练谱系 8 检查点 $\times$ 2 种子 $\times$ 100 = `1,600` 次真实 LIBERO-10 回合，任务 5/5 划分选择集与锁定留出集，Oracle `60.0%`）：人类「直接取最后一个检查点」留出集 `60.00%`（Regret `0.00`）；随机搜索 `54.40%`、Vizier-style TPE `54.33%`、Thompson sampling `54.16%`、人类最佳观测 `54.04%`、$e$-process `53.94%`（误晋升 `0.067`）、固定 8-pt `51.84%`（误晋升 `0.005`）、硬化 12-pt `50.00%`（误晋升 `0.001`；对比原规则 `0.005–0.021`）。
    - 完整性与并发：非 LLM 解析 + HMAC 检出 `6/6` 类伪造（原路径检出 `0`）；围栏扇出通过 `2,000` 次乱序调度零损坏（无围栏在 straggler 试验中损坏率 `99.6%`）；锁定验收集在 `53.9%` vs `70%` 目标下返回 `ACCEPTANCE_FAILED`（并拦截 3 次未授权读取）；全研究消耗 `121.6` GPU-hours（5,475 次真实回合），未记录控制器 token 开销。
- **核查边界：** 全部实测基于 LIBERO-10 仿真，无真机硬件实验（Fig. 1 中 ABB IRB 120 与 YuMi 仅作 CAD 接口示意）；MCP 调用面与工具注册表虽有 `106` 个单元测试，但尚未在真实闭环日志回合中运行过，artifact 过期机制亦未在真实轨迹上触发。
- **对 wiki 的映射：** [You Don't Need To Stay in The Loop: An Agentic Robotics Loop for Robot-Policy Improvement](../../wiki/entities/paper-agentic-2608-07555.md)
