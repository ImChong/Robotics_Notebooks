# ResearchBoost-Agent — 官方代码与验收协议

- **类型：** repo / AI4AI / 自动训练性能优化 / coding agent
- **项目名称：** ResearchBoost-Agent
- **官方仓库：** https://github.com/DAGroup-PKU/ResearchBoost-Agent
- **作者技术主页：** https://ydove0324.github.io/hx_resume/blog/researchboost/
- **核对版本：** 2026-11-08 所见默认 `main` 文档；该版本采用单候选计时 / 最终 B-C-B 计时与单种子 3σ 的当前 judge 规则；历史 `tasks/` 部分代码采用更早的统计验证版本，不得混同。
- **发布记录：** README 记载 2026-10-19 首次公开、11-05 轻量验证调整，11-07 社区案例扩展。
- **代码开放状态：** 公开可访问的训练优化 Agent、数值组件、任务评测脚本和运行入口；仓库根目录在本次核查中 **未见 LICENSE 文件**，不可仅以“公开仓库”推断全部代码获得宽松商业重用授权。依赖的部分 SGLang skill 另有 Apache-2.0 归属说明。
- **入库日期：** 2026-11-08
- **核对范围：** 已通读主仓 README、docs/methodology.md、docs/judge.md、docs/run-a-task.md、作者博客全文；阅读 docs/goal-mode.md 的运行/沙箱/监督关键章节，直接阅读 goal/clawo/run.sh、judge.sh、accept.py、goal/templates/FORMULATOR_BRIEF.md 的配置及入口部分，并核查 tasks/dmd2/bench/bench.py 的注释、配置与主干判定结构，交叉核对 docs/results/embodied-ai.md、docs/results/llm.md。未完整审计每个任务的全部源码、GPU 驱动与环境依赖。
- **通读：** 已完成

## 一手入口

1. [README：总体流程、20 项分域结果、快速开始](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/README.md)
2. [Methodology：问题约束、评测制定、逐轮优化、长程验证](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/methodology.md)
3. [Judge：四级 Gate、check modes 与限制](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/judge.md)
4. [Goal mode：claw-orchestrator / 沙箱 / 监督](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/goal-mode.md)
5. [Run a task：工作目录、基线准备与命令](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/run-a-task.md)
6. [Formulator brief：制定任务及用户决策边界](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/goal/templates/FORMULATOR_BRIEF.md)
7. [run.sh：封装目标、冻结 SHA-256、构造最终验收合约](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/goal/clawo/run.sh)
8. [judge.sh：从 FINAL.json 读取候选，驱动 --final](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/goal/clawo/judge.sh)
9. [accept.py：结果 PASS / 最低加速比 / 判官哈希验收](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/goal/clawo/accept.py)
10. [DMD2 bench.py：已归档的任务特定示例](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/tasks/dmd2/bench/bench.py)
11. [Embodied AI：Motus、RoboTwin、FastWAM、OpenWAM](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/results/embodied-ai.md)
12. [LLM：NanoChat、YOCO、LLM2Vec](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/results/llm.md)

## 已核实的机制

### 冻结之前：所有者定范围，Formulator 定判官

- 所有者确定 upstream commit、工作负载/每卡 batch、精度、训练语义和目标；Formulator 执行上游代码、定基线及数值适配器，并为 `bench` 准备 `time` / `profile` / `check`。
- 对原始代码记录两次 deterministic bitwise baseline，按多个种子冻结 baseline σ；加入负例和学习率扰动 control arm，确认 judge 真能挡住错误；对 judge、阈值及记录求哈希。
- 评测与 formulation 不暴露给优化 Agent 的可编辑路径。bubblewrap 可用时隔离读写；不可用时会进入直接模式，**仅最终哈希检查**保护判官完整性。

### 逐轮优化与验收

- `goal/clawo/run.sh <task> "<GPUS>"` 启动一次 Codex/Claude Code `/goal`，经 `claw-orchestrator` 管理，启动前记录 `judge.sha256`；Agent 逐轮 Profile → 修改一个热点（开关默认关）→ time / quick check → 记 WORK_LOG。
- `FINAL.json` 保留当前最优候选目录和配置。Session 结束后，`goal/clawo/judge.sh` 设置专用最终评测标记，执行 `bench.py check --final`；`accept.py` 读取 `check.json` 验证 `PASS`、`speedup >= MIN_SPEEDUP`、文件哈希不变。
- 当前 `docs/judge.md`：Gate 0 检查入口 AST、数据 batch、RNG 等确切语义；Gate 1 判断加速利益，`--final` 采用 **baseline → candidate → baseline** 独立进程计时；Gate 2 短程 tensor bitwise；不逐比特相同则进入 Gate 3 冻结种子的一条配对轨迹，各端点偏差需处于 **3σ** 范围。
- `check --quick` 不等于完整 PASS；`--final` 与生产训练完整质量验证也不等价。部分 `tasks/` 存档判官代码仍显示旧的“两至六种子/2σ”设计，应以对应任务原始评测日志和版本解释，不把所有历史案例直接套当前门槛。

## 原始结果与口径（作者公布，未独立复测）

| 场景 | 硬件 | 性能结果 | 判定和测量范围 |
|---|---|---|---|
| Faster-Motus / RoboTwin 2.0 微调，27,244 episodes | 8× B200 | **9.59×**；7.468→0.779 s/iter | 1,100 iterations、跨两个 epoch 边界、8 ranks 逐比特一致；报告**单次迭代**时间 |
| Faster-NanoChat / D24 ≈1.384B 参数 FP8 训练 | 8× H20 | **1.0744×**；3358.4→3126.0 ms/update | 短程统计等效；**不包含启动/编译/保存/完整训练验证** |
| Faster-RoboTwin / π0.5 policy evaluation | 8× H20 | **4.38× (estimated)** | 73.5→16.8 h 全量任务耗时从部分作业外推，并非整套实测 |
| Faster-FastWAM / LIBERO | 8× B200 | **3.46×**；1541.4→445.7 ms/iter | 作者提供 LIBERO 评测对照，指标不能混同单步 bitwise |
| OpenWAM-α SFT | 4× H200 | **2.54–2.77×** | 七个 benchmark、与原 main 比较；包含改变 checkpointing 策略的收益 |

- **20 个项目**是 README 2026-11-08 汇总的跨 7 个领域案例数量；不是统一评测排行榜，也不是一套训练任务的平均加速。
- [Faster-Motus 的原始记录](https://github.com/AI4AI-for-Anything/Faster-Motus/blob/main/docs/full-data-results.md)
- [Faster-NanoChat 的原始记录](https://github.com/AI4AI-for-Anything/Faster-NanoChat/blob/main/docs/training.md)

## 重要边界

1. `--final` 判官验证**被选的负载与检查内容**，不能证明任何长度、任务或数据集上的训练都等价；短轨迹不一定捕捉缓慢漂移。
2. 单步加速不包含初始化、数据预处理、编译、checkpoint、外部评价、机器人真机实验等固定成本；应在 real workload 上完成端到端验证。
3. 并非每个案例都只含纯数学等价：例如 OpenWAM 的 checkpointing 路径变化，收益应连同精度与显存条件理解。
4. 安全隔离取决于 `bubblewrap` 权限/主机环境；README 的“自动化”不等于 Agent 可以无成本、无权限问题接管任意 GPU 训练集群。
5. 本次是官方文档与关键源码静态核验；没有实机训练，不主张独立证实官方加速比。

## 对 wiki 的映射

- [ResearchBoost-Agent：冻结判官驱动的训练代码自动优化](../../wiki/entities/researchboost-agent.md)
