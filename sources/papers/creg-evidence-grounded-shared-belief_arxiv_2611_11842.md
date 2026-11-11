# CREG：互联机器人证据落地共享信念更新基准（arXiv:2611.11842）

> 来源归档（paper）

- **标题：** A Benchmark and Layered Evaluation Protocol for Evidence-Grounded Shared Belief Updating in Connected Robotics
- **类型：** paper
- **作者 / 机构：** Ranulfo Bezerra、Kimitoshi Yamazaki（Tohoku University）；cs.RO，v1 日期 2026-11-09，页眉标 PREPRINT UNDER REVIEW
- **arXiv：** <https://arxiv.org/abs/2611.11842>
- **PDF：** <https://arxiv.org/pdf/2611.11842v1>
- **项目页：** <https://creg-bench.github.io/>（下载页：<https://creg-bench.github.io/download/>）
- **代码：** https://github.com/RanulfoBezerra/Evidence-Graph-Toolkit （项目页 Code 链接与下载页 `evaluation-v1.0` 标签指向此仓库）
- **入库日期：** 2026-11-11
- **一句话说明：** 350 个确定性生成的合成场景（7 个 regime）+ 每个证据前缀的规范后验参考 + 分层评测协议，用来评「证据到信念的轨迹」而非只评最终答案。
- **核对范围：** 读了 arXiv v1 PDF 正文 I–VII 节、全部表（Table I–IV）与图注（Fig. 1–6）、附录 A–I 全部文字与表（A1–A7）；参考文献列表只扫过未逐条核对；图为 pdftotext 抽取的文字与图注，未逐像素看图。项目页首页与下载页的 HTML 文字已读；GitHub 仓库页面在本环境返回空壳，README 与代码**未核实**，也未下载数据 / 未运行任何代码。
- **通读：** 已完成

## 开源状态

- 项目页下载页列出三份冻结文件（scenario JSONL 约 6.7 MiB、manifest、normative spec，均带 SHA-256）与 git 标签 `evaluation-v1.0` 的工具包（生成器、独立 log-odds 检查器、提示模板、九种表示渲染器、指标、统计、全部图表脚本）；仓库许可证具体内容未核实。
- 论文附录 I：每场景的原始模型输出与统计工作区以 release artifact 形式分发，不在源码仓库里；模型 manifest 的 model_revision / tokenizer_revision 字段未设置，精确复现依赖当时同名模型快照。

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **基准构成（Sec. III-A, Table A4）：** 350 场景 = 7 regime × 50（justified revision、belief preservation、relation association、contradiction onset、contradiction resolution、stale/delayed evidence、source reliability）；每个场景一个目标实体、二值物理状态（anomalous / normal）、均匀先验；共 1,150 个场景-前缀对。有效准确率 = 0.5 + (confidence − 0.5) · reliability。
2. **校验（Sec. III-A/B, App. F）：** 独立 log-odds 检查器在全部 1,150 个前缀上与生成器一致；1 位专家对 35 个场景-前缀对做盲评，第一轮接受 29/35，分歧源于并列后验语义的规范问题，修规范后重新生成，第二轮 35/35；作者自述这只是语义合理性检查，不代表人类共识，无评分者一致性统计。
3. **表示条件（Sec. III-C, Fig. 3）：** canonical JSON、structured list、full graph、graph without relations 四个主条件，外加仅 Qwen2.5-7B 的 5 个因子消融（flat edge-removed、no provenance、no timestamps、no confidence、shuffled graph）。已披露的混淆：canonical JSON 提示长度为 full graph 的 0.843 倍；no-provenance 同时删掉了 source-reliability 字段。
4. **模型与协议（Sec. IV-A, App. B）：** Qwen2.5-1.5B、Qwen2.5-7B、Mistral-7B-Instruct-v0.3、Gemma-3-12B-IT；采样、温度 0.2、top-p 0.9，3 个生成种子，512 token 输出上限。因算力上限（预注册 48 h）触发回退：所有模型只跑 canonical JSON，完整表示 / 因子矩阵只给 Qwen-7B（共 12 个系统）。Gemma 在其余三种表示上的运行是探索性的。另有 6 个透明融合基线。
5. **结果 1（Sec. V-A, Table II/III）：** 规范 JSON 下实体落地率 0.694（Mistral-7B）到 1.000（Qwen-7B）；轨迹得分 Qwen-1.5B 0.397、Mistral-7B 0.239、Qwen-7B 0.797、Gemma-12B 0.801；失败几乎全是「缺实体」而非「幻觉实体」（幻觉率 ≤ 0.016）。仅在已落地前缀上的条件得分（探索性、非预注册、上界）：Qwen-1.5B 0.747、Mistral-7B 0.755。作者明确不结论落地是唯一缺口来源。
6. **结果 2（Sec. V-B, Table III, Fig. 5）：** 在 relation-association regime 里删除语义关系使 Qwen-7B 严格信念态准确率降 15.3 个点（CI [0.098, 0.207]，Holm p < 0.001）；同一 regime 的轨迹得分差仅 +0.020（CI [0.002, 0.037]，Holm p = 0.058，不显著）；全体 regime 的轨迹得分差 +0.088。信息匹配的 structured list 0.842 > full graph 0.814；打乱记录顺序的 shuffled graph 0.848 > 有序 graph（+0.034）；flat edge-removed 0.774 < full graph。作者结论：关系内容有用，图序列化本身无一致优势。
7. **结果 3（Sec. V-C, Fig. 6）：** Qwen-7B 的图族条件在第 3 个证据前缀的信念态准确率落到 0.20–0.25，第 4 个前缀回升到 0.62–0.81；canonical JSON 最低点 0.64（同在第 4 前缀）；Mistral-7B 的落地率在第 1–4 前缀分别为 0.009、0.051、0.860、0.470。仅看最终前缀会掩盖这些中途失败。
8. **透明基线与动作（Sec. V-D, Table IV, App. D/E）：** 六个融合基线轨迹得分 0.603（latest-observation）到 0.700（source-reliability weighted），均高于两个诊断层模型、低于全部核心神经系统（0.727–0.848）。动作任务：精确参考策略一致率 0.214 / 0.213 / 0.215 / 0.149（Qwen-1.5B / Mistral-7B / Qwen-7B / Gemma-12B），证据支持动作率四者都是 0.857（结构性相同，区分力有限）；确定性规则流水线选中规范动作 125/350（35.7%）。始终覆盖模型提议的确定性策略 harmful-override 率 0.494（Table A5）。
9. **自述局限（Sec. VI-D, App. G/I）：** 全部合成；二值状态、单目标实体；物理日志重放只是 2 条观测的样例管线演示（distributed_gauge_inspection），不是机器人评测；单专家审计；两个表示消融存在混淆；时间戳操控刻意偏弱；无独立 source-reliability 消融；条件轨迹得分与 Gemma 非规范运行为探索性；只测 1.5B–12B 四个开源模型、512 token，更大模型、工具增强与更长预算未测。

## 对 wiki 的映射

- [paper-creg-evidence-grounded-shared-belief](../../wiki/entities/paper-creg-evidence-grounded-shared-belief.md)
