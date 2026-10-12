# MARGoS 论文来源归档

## 元数据

- **题目：** MARGoS: A Configuration-Driven Platform for Reproducible MARL Experiments in ARGoS
- **作者：** Tom Philipp Backert, Mohammad D. Soorati, Javad Ghofrani
- **单位（HTML）：** University of Lübeck；University of Southampton；Bonn-Rhein-Sieg University of Applied Sciences
- **预印本：** https://arxiv.org/abs/2610.13772 （v1，2026-10-09，cs.SE / cs.MA / cs.RO）
- **HTML：** https://arxiv.org/html/2610.13772v1
- **正式出版：** VARIABILITY 2026 Project Showcase Track，LNCS 17104 pp. 251–261，DOI 10.1007/978-3-032-40421-3_24
- **代码：** https://github.com/tombackert/margos （MIT，已开源）
- **核对日期：** 2026-10-12

## 论文要解决的问题

ARGoS 常用于群体机器人仿真，多智能体强化学习（MARL）又让实验由模拟器文件、训练脚本、配置值、路径、种子和环境细节手工对齐拼成，难以复用，课题组内部交接都脆弱。论文把这些「定义实验的输入」变成平台托管的产物。

## 方法与系统信息

- **流水线：** 配置 → 编排 → 日志 → 分析 → 导出/导入；命令行仅五条：`run`、`show`、`compare`、`export`、`import`（没有独立 reproduce 命令：复现 = import 后 run，再用 compare 对照参考）。
- **配置哈希：** 对规范化冻结配置计算；配置只存文件引用不存内容，所以哈希不是源码版本管理，重跑同一性默认被引用源文件未变。
- **种子：** 端到端传播，并有运行时配置完整性校验；**环境指纹**随每个完成的 run 记录并进入导出包。
- **实验包：** 含冻结配置、完整性哈希、指纹、日志，以及场景与训练脚本源码副本；导入时引用重写到包内副本。
- **桥接：** ArgosToZoo 通过 ZeroMQ 把 ARGoS（C++）接到 PettingZoo 兼容环境，训练脚本使用 RLlib。

## 论文报告的结果（五维评估）

| 维度 | 设计 | 结果 | 目标 |
|---|---|---|---|
| RQ1 架构 | 协议可用性 | 平台端到端跑通 RQ2–RQ5 协议 | 仅验证可用 |
| RQ2 效率 | 每条件 N=5，单专家评估 | 时间 87.0±15.02 s → 19.0±1.41 s（−78.2%）；步数 6 → 2（−66.7%） | ≥50% |
| RQ3 复现 | 同硬件 20 次自动重跑 | 20/20 与参考一致，SD 0.0000 | ≥90% |
| RQ4 可用性 | 35 条启发式审计 + KLM | 30/35 合规（85.7%）；七任务加权 KLM 缩减 56.9% | ≥80%；≥50% |
| RQ5 协作 | 手工与平台各 10 次交接 | 交接成功率两者均 100%；分享步数 8 → 1（−87.5%）；分享时间 −65.3%；复现时间 −45.9%（略低于 50% 目标）；首次运行慢 7.4% | ≥50%；≥90% |

KLM 细节：T3–T7（执行、监控、结果、导出、导入/运行）缩减 58–88%，配置编辑 T1–T2 为 0%（两种条件用同一编辑器）。结论：RQ2–RQ4 获支持，RQ5 部分支持。

## 复现与解释边界

- 单个开发者兼评估者，存在自评偏差；KLM 模拟专家无错时间，不等于真实用户的节省。
- 复现只在同一硬件（macOS，CPU）验证，跨硬件/跨实验室未验证；协作者使用已准备好的账户，存在可测量的库漂移。
- 只评估了 ARGoS+RLlib 栈和实际执行的任务集；阈值与效应量判断而非总体统计推断。
- 预印本未经同行评审，正式版为 LNCS 论文。
- 本质是研究工程基础设施，不含新的 MARL 算法或机器人性能结果。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-margos-reproducible-marl-argos.md
- 仓库归档：sources/repos/margos.md
