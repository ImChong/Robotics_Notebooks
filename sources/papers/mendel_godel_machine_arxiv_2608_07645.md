# Mendel Gödel Machine：比较进化的编码智能体递归自我改进

- **英文题目：** Mendel Gödel Machine: Recursive Self-Improving Coding Agents via Comparative Evolution
- **作者：** Changzhi Liu、Yilun Liu、Sikuan Yan、Volker Tresp、Yunpu Ma
- **机构：** 电子科技大学、慕尼黑大学（LMU Munich）、慕尼黑机器学习中心（MCML）
- **arXiv：** [2608.07645v1](https://arxiv.org/abs/2608.07645v1)，2026-08-07
- **正文：** [HTML 全文和附录](https://arxiv.org/html/2608.07645) / [PDF](https://arxiv.org/pdf/2608.07645)
- **项目：** https://reallcz.github.io/MGM/
- **官方代码：** https://github.com/RealLcz/MGM
- **核查日期：** 2026-11-20
- **核对范围：** 阅读 arXiv v1 §1–7、附录 A–H 的算法、理论假设、实验与局限；对照官方项目页全部主要方法、结果与开源入口，并检查代码 README、hgm.py、hgm_utils.py、self_improve_step.py、tree.py、config.yaml 和 Slurm 启动脚本。
- **通读：** 已完成

## 摘要

MGM 沿用 Huxley Gödel Machine（HGM）的开放式编码 Agent 进化树，将过去只基于一条失败日志的自修改扩展为利用存量任务轨迹的比较证据，不更新底座语言模型的参数。

## 机制（论文 §3、附录 A）

1. **CM（Clonal Mutation）：** 单节点、单失败任务的传统自修改。
2. **RM（Reaction-norm Mutation）：** 同一个节点的不同任务结果联合诊断；需要多个评测结果。
3. **CH（Cross-lineage Hybridization）：** 对比不同谱系在同一任务上的执行结果；一个成功一个失败时编辑失败侧、成功侧充当参考，不直接拼接两者的代码；双方均失败时以整体 utility 确定 Primary。
4. **调度：** 以 Thompson sampling 分配新增评测（φ-evaluation）和新增后代（Φ-expansion）的预算。三算子权重在主要实验中为 0.10:0.45:0.45，只有满足资格时才参与归一化抽样。
5. **理论边界：** 加性适应度模型和比较诊断信息的有效性假设成立时，比较策略有利于搜索；不能当作所有任务的形式化收益证明。

## 论文报告的实验（§5）

| 基准 | 初始 Agent | HGM | MGM |
|---|---:|---:|---:|
| SWE-bench Verified-60 | 68.3% | 73.3% | **78.3%** |
| Polyglot-60 | 50.8% | 77.9% | **93.2%** |

主要实验使用 Qwen3.6-35B-A3B、200 次 φ-evaluation、24 次 Φ-expansion；完整 Polyglot-225 的额外评测是 **210/225=93.3%**，与 Polyglot-60 的 93.2% 不可混为一谈。论文报告跨基准、跨模型迁移，均为编码 Agent 任务，不是真机机器人研究。

## 复现及开源边界

- 项目主页提供论文和 GitHub 代码；README 提供 Linux / Python 3.11、Slurm 或直接运行、vLLM/Apptainer 等使用指导。
- 当前 hgm_utils.py 的 sample_child() 还调用 Docker 客户端，与 README 所述“无需 Docker daemon”存在运行环境版本/模式差异；部署应先核查。
- config.yaml 的 failed_pool_boost 为 1.0（默认无失败任务附加权重），不要将选项写成默认已开启；README 示例与当前 config.yaml 的模型默认值也不同。
- 本次未重跑模型推理、复杂编程基准或大型进化过程。
- 用户同时提供的 [arXiv:2407.19594v2](https://arxiv.org/abs/2407.19594v2) 实际是 **Meta-Rewarding Language Models**，不是 MGM 原论文，该文此前已独立入库。

## 对 wiki 的映射

- [MGM 独立论文实体](../../wiki/entities/paper-mendel-godel-machine.md)
- [DGM 对照](../../wiki/entities/paper-darwin-godel-machine.md)
- [RSI Stage 4](../../roadmap/depth-rsi.md)
- [MGM 源码归档](../repos/mendel-godel-machine.md)
- [MGM 官方项目页](../sites/mendel-godel-machine.md)

## 参考来源

- [原论文 HTML 及附录](https://arxiv.org/html/2608.07645)
- [原作者项目主页](https://reallcz.github.io/MGM/)
- [源码仓库](https://github.com/RealLcz/MGM)
