# PRM-as-a-Judge 1.5: A Toolkit for Robot Process Assessment（arXiv:2608.14284）

> 来源归档（ingest）

- **标题：** PRM-as-a-Judge 1.5: A Toolkit for Robot Process Assessment
- **缩写 / 框架：** **PRM-as-a-Judge 1.5**（Process Reward Model as Judge）
- **类型：** paper / evaluation / progress-reward / vla / wam / toolkit
- **arXiv：** <https://arxiv.org/abs/2608.14284>（PDF：<https://arxiv.org/pdf/2608.14284>）
- **1.0 论文：** <https://arxiv.org/abs/2603.21669>（仓库 README 主引用；1.5 为指标与评测套件升级）
- **项目页：** <https://prm-as-a-judge.github.io/> — 归档见 [`sources/sites/prm-as-a-judge-github-io.md`](../sites/prm-as-a-judge-github-io.md)
- **代码：** <https://github.com/Yuheng2000/PRM-as-a-Judge> — 归档见 [`sources/repos/prm-as-a-judge.md`](../repos/prm-as-a-judge.md)
- **作者：** Yuyang Liu\*、Yanqing Shen\* 等；Project Lead Yuheng Ji；通讯 Pengwei Wang（BAAI）、Xiaolong Zheng（CASIA）
- **机构：** 北京智源人工智能研究院（BAAI）；中国科学院自动化研究所（CASIA）
- **入库日期：** 2026-08-17
- **一句话说明：** 把 rollout 视频经 PRM 变成进度曲线，再用 OPD 指标（含 FNS / DRR / SQS）做过程级评测，并配套 RoboPulse++ 检验评测器本身。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** 首页链 Paper / Blog / Leaderboard / User Guide；GitHub 工具仓已上线；默认 judge 权重 [Robo-Dopamine-GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview)；[RoboPulse](https://huggingface.co/datasets/yuheng2000/RoboPulse) 已上线；**RoboPulse++ 徽章仍为 Coming Soon**。
- **仓库核查：** Apache-2.0；可运行入口 `eval/run_eval.sh`、`eval/run_judge.py serve`、`getting_started/PRM_as_a_Judge_quickstart.ipynb`。
- **结论：** **已开源（评测套件 + 可视化）**；RoboPulse++ 数据发布仍待齐。

## 摘录 1：问题与主张（§1）

- **痛点：** 操纵榜被二元成功率或手工规则分主导；失败轨迹之间、成功轨迹之间的过程质量被压扁。
- **主张：** 只给任务描述 + rollout 视频，即可得到进度曲线与 Outcome–Process–Diagnosis（OPD）报告。
- **1.5 相对 1.0：** 保留 OPD，新增条件指标 FNS / DRR / SQS；在 RoboDojo 冻结榜（2026-07-03）上评一批 VLA / WAM；引入 RoboPulse++ 评 PRM 本身。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-prm-as-a-judge.md`](../../wiki/entities/paper-prm-as-a-judge.md)；回链 [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md)、[RoboDojo](../../wiki/entities/robodojo.md)、[评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)。

## 摘录 2：方法与指标（§2、附录 B）

- **管线：** JSONL manifest（case_id / task / video）→ PRM 出 \(p_{0:T}\in[0,1]\) → 平滑后算 OPD → 交互报告。
- **默认 judge：** Robo-Dopamine（Forward）；pair-style 也可 Incremental / Backward。
- **Outcome：** MP、MC@\(q\)（25/50/75）。
- **Process：** \(\mathrm{PPL}=\mathrm{MP}^2 / \sum |p_t-p_{t-1}|\)。
- **Diagnosis：** CRA、STR；1.5 新增 FNS（失败近成功）、DRR（回撤恢复）、SQS（成功质量）。

**对 wiki 的映射：** 实体页用指标表写「SR 不够用」的读法，不抄公式全集。

## 摘录 3：评测与发现（§3–4、附录 C）

- **被评对象：** RoboDojo-RealWorld（\(\pi_{0.5}\)、InternVLA-A1、Xiaomi-Robotics-0、GalaxeaVLA、X-VLA、\(\pi_0\)、StarVLA-\(\alpha\)、GR00T-N1.7、Spirit v1.5 等）；RoboDojo-Sim 另含 Hy-Embodied-0.5-VLA、X-WAM、Fast-WAM、LDA-1B 等。
- **关键发现：** SR 排名与 SQS / DRR / FNS 不完全一致；Sim 上 VLA 整体强于 WAM；更大参数不保证更好；\(\pi_{0.5}\) 多指标最稳；Precision 任务相对最好、开放词汇最难；Sim–Real 排名相关弱（Spearman \(\rho=0.18\)–\(0.58\)）。
- **RoboPulse++：** 700 轨迹 / 275 任务 / 2,244 区间，标 Rising / Falling；Robo-Dopamine (Forward) Macro-F1 **0.77**、Acc **0.84**，优于通用 VLM；Falling 仍难（最佳 F1 0.63 vs Rising 0.92）。

**对 wiki 的映射：** 把「过程指标打乱 SR 榜」与「评测器自己也要测」写进结论。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-prm-as-a-judge.md`**、**`sources/sites/prm-as-a-judge-github-io.md`**、**`sources/repos/prm-as-a-judge.md`**。
- 交叉更新过程奖励、RoboDojo、评测选型闭环。
