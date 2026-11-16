# Seeing Through the Illusion: Test-Time Scaling for World Action Models（arXiv:2611.03585）

> 来源归档（paper）

- **标题：** Seeing Through the Illusion: Test-Time Scaling for World Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.03585>（v1，2026-11-02，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.03585>
- **作者与机构（首页）：** Cheikh Ahmed、Mahdi Mostajabdaveh、Zirui Zhou（Huawei Technologies Canada）；Milad Yazdani（University of British Columbia）
- **代码：** 论文正文未给出代码或项目页链接（未核查论文之外的项目页，截至入库日 2026-11-09 未确认开源状态）
- **入库日期：** 2026-11-09
- **一句话说明：** 提出 WISE：对冻结 WAM 采样的 B 个「动作 + 想象未来」候选，在执行前按动作有效性、IDM 一致性、任务进度三项打分并执行得分最高者；B=4 时 RoboLab 成功率 6.0%→24.0%，LIBERO 平均 75.1%→78.3%。
- **核对范围：** 下载 arXiv PDF（首页标题与编号 2611.03585 一致），用 pdftotext 抽取全文并逐段读完正文第 1–6 节（摘要、引言、相关工作、问题定义、方法 4.1–4.4、实验 5.1–5.4、结论）、表 1–7、参考文献 [1]–[27]；全文共 11 页，无附录或补充材料。图 1、图 2 仅读了正文与图注及图内文字（未逐像素看图）。未读被引论文（Robometer、GeoBoN、Future Consensus、DreamZero 等）原文，未运行任何代码。
- **通读：** 已完成

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **问题（§1、图 1）：** WAM 联合生成动作与想象未来，但「合理的预测不一定被其共生成的动作实现」；图 1 给出 RoboLab 上 Cosmos 3 Edge 的两个 dream-execution 分歧例子（香蕉抓取、酸奶移动）。作者以「假设」表述：动作与想象未来不一致的候选最可能在执行中偏离，且这种不一致可在执行前度量。
2. **方法（§3、§4）：** 每个控制步冻结 WAM 采样 B 个候选 c=(X̂, A)（式 1），选 R_WISE 最大者（式 2），R=r_exec + r_cons + r_task（式 3），执行整段动作块后重规划。
   - r_exec（§4.2，式 4–5）：位置 / 速度 / 加速度 / jerk 等检查的最坏违例经 exp(−δ/τ) 软化，加权平均。
   - r_cons（§4.3，式 6–10）：IDM 从想象未来恢复动作，与 WAM 动作比较；连续通道用标准化 L1 差 exp(−L)，离散通道（夹爪）用阈值后一致率，二者各占 1/2。
   - r_task（§4.4，式 11）：冻结 Robometer 对最终想象帧的进度分（[0,1]）。
3. **设置（§5.1）：** RoboLab 上 Cosmos 3 Edge，10 个任务 × 5 episode = 50 episode；LIBERO 上 DreamZero（公开 RLinf WAN2.2-5B LIBERO SFT checkpoint，step 26,000），四个 suite 各 500 episode，共 2,000。基线：Future Consensus、GeoBoN、Robometer，B=4；直接执行 B=1。IDM 分别在 DROID（21,000 训练 / 1,000 验证 episode）与 physical-intelligence/libero 演示（1,693 episode，5% 验证）上训练；RoboLab IDM 把 33 帧 rollout 映射到 32 步动作块，LIBERO IDM 由 9 帧双视角解码片段预测 16 步动作块。
4. **主结果（§5.2，表 2–4）：** RoboLab：6.0 → WISE 24.0，Future Consensus 18.0、GeoBoN 20.0、Robometer 20.0。LIBERO 平均：75.1 → 78.3（Spatial 92.6→95.8，Object 88.2→93.0，Goal 51.6→53.6，Long 67.8→70.8）；GeoBoN 在 Goal（55.0）、Robometer 在 Long（72.2）领先；WISE 比 Robometer 平均高 0.9，作者称约一个标准误。耗时：相对 B=1，WISE 3.4×（RoboLab，3062.5 vs 892.5 s/ep）、3.5×（LIBERO，25.2 vs 7.1 s/ep）。
5. **消融（§5.3，表 5–7）：** RoboLab 单分量选择器：Exec-only 10.0、Cons-only 16.0、Prog-only 20.0，完整 24.0。三项分量同时偏好同一候选仅 7.8%（RoboLab）/ 7.0%（LIBERO）；去掉任务进度分会改变约 47.1% / 48.5% 的选择，为三者中最大。
6. **作者自述局限（§5.4）：** 依赖辅助评估器（IDM 可能误恢复动作，进度模型可能误判视觉上合理的未来）；单分量实验不衡量「在另两项存在时的增量收益」；RoboLab 每方法仅 50 episode，数个百分点的差距在抽样波动内；推理成本随候选数成倍增加，WISE 再加 IDM 与进度模型开销。
7. **未来工作（§5.4、§6）：** 自适应重规划、把 WISE 分数当 WAM 后训练奖励；更大样本量的带种子评测、成对分量消融。

## 与周报说法的核对（周报「具身运控lab」条目，二手资料）

- 机构：周报写「待核实」；论文首页为 Huawei Technologies Canada（前三位）与 University of British Columbia（Yazdani）。
- 「检查动作有效性、时间连续性、逆动力学一致性和任务进度」：论文是三个分量，动作有效性与时间规整性合为 r_exec（§4.2），并非四个独立评分项。
- 「四候选设置下成功率从 6.0% 提高至 24.0%」：数字与摘要、表 2 一致，但仅是 RoboLab（Cosmos 3 Edge，50 episode）；周报未提 LIBERO 的 75.1%→78.3% 与相对最强基线仅 +4.0 / +0.9。
- 「保持 WAM 冻结，只选择评分最高的动作」：准确，需补充执行的是整段动作块，且 IDM 需按本体单独训练（§1、§5.1）。

## 对 wiki 的映射

- [paper-wise-wam-test-time-scaling](../../wiki/entities/paper-wise-wam-test-time-scaling.md)
