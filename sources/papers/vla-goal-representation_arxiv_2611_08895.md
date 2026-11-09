# Localizing and Editing the Goal Representation of a Vision-Language-Action Model（arXiv:2611.08895）

> 来源归档（paper）

- **标题：** Localizing and Editing the Goal Representation of a Vision-Language-Action Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08895>（读取版本 v1，2026-11-06 提交，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.08895v1>
- **代码：** <https://github.com/sudsho/vla-goal-representation>（Zenodo DOI 10.5281/zenodo.23871372，见论文附录 A 与仓库 README）
- **作者 / 机构：** Sudhanshu Srivastava、Faisal Aqlan，University of Louisville（论文首页）
- **发表：** SPAIS Workshop @ CoRL 2026（论文首页脚注）
- **入库日期：** 2026-11-09
- **核对范围：** 读了 arXiv v1 PDF 全部 8 页：摘要、§1 引言、§2 相关工作、§3 方法、§4 实验（含图 1、图 2 的图注与表 1）、§5 局限、参考文献、附录 A（实验细节）；仓库 README 静态阅读（未运行任何代码，未下载 Zenodo 数据、未逐文件读源码）。图中曲线以图注与正文数字为准，未做逐点读数。
- **通读：** 已完成
- **一句话说明：** 在 LIBERO-Goal 单场景中审计 OpenVLA-7B 对禁止类指令（"do not X"）的遵从情况，用激活修补、线性探针、激活转向定位并编辑目标表示，并比较「目标读出」与失败检测器作为运行时监控的效果。

## 开源状态

- **已开源**（论文附录 A 给出 GitHub 与 Zenodo；2026-11-09 读 README 核对）：仓库含实验计划、逐 episode 摘要、分析输出、转向向量，以及复现数字 / 表 1 / 图的 CPU 脚本；原始 rollout（约 29 GB 激活与帧）**未包含**，GPU 实验需自行下载 `openvla/openvla-7b-finetuned-libero-goal` 并安装 LIBERO（README 所述，未实际运行）。

## 核心摘录

1. **设置（§3、§4）：** OpenVLA-7B 在 LIBERO-Goal 微调，贪心解码；名义成功率 78%（95% CI [73, 82]，352 个 rollout），与已发表的 79.2% 接近。否定条件 120 个 rollout（5 种禁止措辞，每条禁止一整个任务）；重定向（redirect）条件 285 个 rollout、6 个任务对。执行目标 y 由 10 个任务的目标谓词在每一步判定，持续 w=5 步成立即记。
2. **否定无效（§4）：** 否定指令下执行被禁任务 95/120（79%，CI [71, 85]），其余 25 个停滞；配对的 92 个情形中，被禁任务执行 76/92、普通指令成功 81/92（McNemar p=0.30）。附录 A：五种措辞分别为 27/30、27/30、12/20、13/20、16/20；配对差 −5 个百分点（Newcombe CI [−14, +3]）。重定向仅 35% 被遵从，285 个中 163 个停滞、23 个完成了错误目标。
3. **定位（§4，图 1）：** 修补显示层 12 到 16 之间有突变：层 8 指令 token 决定首个动作（ρ=0.94），层 12 约三分之二（0.69），层 16 起很小（0.18）；决策 token（最后一个 prompt token）层 8 几乎不起作用（0.04），层 16 起占主导（0.77）。"not X" 与 "X"、"please X" 下目标读出 p(X) 至层 28 都不低于 0.73，即 "not" 没有压低读出；"not" 本身可被线性探测（平衡准确率层 4 为 1.00、层 16 为 0.87、层 28 为 0.76）。
4. **编辑（§4，图 2a）：** 层 8 指令位置的固定转向向量闭环重定向 12/18（67%，CI [44, 84]）；层 12–20 的固定向量与预填充（prefill）修补共 0/158；层 16 以后只有对每个已解码动作 token 都修补（decode-stream patch）才有效：层 16 为 7/18、层 24 为 15/18（83%，CI [61, 94]）。预填充修补在层 16 只传递首维 0.95、其余五维至多 0.04。
5. **监控（§4，表 1，图 2b）：** 否定下，目标读出标出全部 95 次违规执行，但对停滞的 25 次也全标（FA 1.00），作者称其不优于「not」关键词规则；熵只标出 0.21。重定向未遵从检测：层 32 目标探针对内（within-pair）AUROC 0.88 [0.81, 0.95]，成功探针 0.74，动作熵 0.59；外部 DINOv2+词袋探针 0.87，与层 32 探针差 +0.02（CI [−0.02, +0.06]），作者据此称内部访问无可检测增益；对「任务对 + 时间」先验（0.76）的差 +0.13，CI [−0.01, +0.26] 含 0。
6. **作者自述的局限（§5）：** 一个 VLA、一个仿真场景、温和的整任务禁止、仿真器目标标签；多数闭环编辑仅 18 个 rollout、3 个任务对，0/18 对应编辑成功率最高可达 18%（95% CI）；换其他 VLA 与真机之前不应用于部署决策。附录补充：检查点只在 10 条不含否定的 LIBERO-Goal 指令上微调；未测试限制「做法」的禁止；类均值向量的计算包含了测试初始状态。

## 对 wiki 的映射

- [paper-vla-goal-representation-editing](../../wiki/entities/paper-vla-goal-representation-editing.md)
