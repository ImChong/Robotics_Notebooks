# RSIJudge：Learning the Task or Learning the Verifier?（arXiv:2611.08318）

> 来源归档（paper）

- **标题：** Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08318>
- **PDF：** <https://arxiv.org/pdf/2611.08318v1>（v1，2026-11-05，cs.AI）
- **项目页：** <https://dagroup-pku.github.io/PyRUA-Lean/RSIJudge/>
- **代码：** <https://github.com/DAGroup-PKU/PyRUA-Lean/tree/RSIJudge>（RSIJudge 分支；README 声明 Apache-2.0，含循环、忠于指令判据 `faithful_criteria/`、评测与迁移配置）
- **作者与机构（以论文首页为准）：** Wenbo Huang、Jianxin Bi、Shunyu Yang、Ruiyang Si、Rui Ni、Qiang Wang、Shulong Jiang、Yintianrun Zhang、Feixiang Ruan、Kewei Zhang、Ye Huang、Shukai Gong、Daquan Zhou（通讯）；北京大学；Jianxin Bi 另属新加坡国立大学
- **入库日期：** 2026-11-09
- **来源线索：** 具身智能小站 2026-11-09 更新（条目「06 RSIJudge」）
- **核对范围：** 下载 arXiv v1 PDF 并用 pdftotext 逐段通读正文第 1–6 节（含图 1–8 的图注与图中数字）、参考文献、附录 A–J（协议与设置、任务表、判据与指令对照表 5/6、双判据逐任务表 7、读 C^faith 的表 8、记忆条目分析、库迁移与案例表 9、跨基准迁移原因、基元对齐表 10）；另静态读 GitHub 分支 README 与项目页 HTML 摘要；未运行任何代码，未读仓库内 `faithful_criteria/`、`configs/`、`docs/REPRODUCE.md` 等子文件，未逐项复算附录表中的数字，图中曲线只读图注与图内标注数值。
- **通读：** 已完成
- **一句话说明：** 同一自我改进循环只切换「学习者能否读基准判据代码、分数与阶段」（Criteria Exposure，CE）：读判据在原判据下得分更高，但在只验证指令所述内容的新判据下领先消失，且经验迁移到新任务和新基准时没有额外收益。

## 论文要点摘录（均带出处）

1. **设置（§3.1，Algorithm 1，图 2）：** 冻结执行者 GPT-6 Astra（高推理强度），经 PyRUA-Lean 基元写 Python 代码控制机器人；学习者 Claude Code + Opus 5.5（高推理强度）每轮对「技能 + 记忆」库做一次修订，批次明显退步（少 ≥2 次成功）则回滚；每任务独立库，五轮、五个开发布局。唯一开关：w/o CE 学习者只见每回合成败，w/ CE 另见判据代码、分数与阶段。
2. **判据受限任务（§3.2，附录 B）：** 判据与指令在某些轨迹上不一致的任务。RoboDojo 42 个任务中 19 个不一致，加上判据有 bug 的 solve_equation 共 20 个判据受限任务、26 个变体；33 个任务要求双臂回到 home 而只有 5 条指令要求复位。类型名为作者自拟（隐藏规则、硬阈值、时序/不可逆事件、判据缺陷、额外终态、宽松判据、矛盾）。
3. **忠于指令的判据 C^faith（§3.3，附录 C，表 6）：** 只改 run_reward 与 get_score，不改指令、不动原仓库；两套判据对同一批回合独立判定，所有结论翻转的回合人工核对，未发现新判据误判。C^faith 由作者自己编写，仅 RoboDojo 做了改写。
4. **原判据下的结果（§4.1，表 1，图 3）：** 25 个学习任务（16 判据受限 + 9 忠实），测试为 seed 0 的 20 个布局。全部任务：空库 24.6%，w/o CE 47.8%，w/ CE 62.0%。判据受限：8.4 / 27.8 / 51.6；忠实：53.3 / 83.3 / 80.6。开发曲线：判据受限任务两者第 0 轮均 6%，末轮 w/o CE 31%、w/ CE 55%；忠实任务末轮 91% 与 95%。
5. **换判据重判（§4.2.1，图 4，表 7）：** 16 个判据受限任务各 320 回合。C^orig 下 w/ CE 51.6% 对 w/o CE 27.8%；同一批回合用 C^faith 判，45.0% 对 47.2%；差距缩小 25.9 个百分点（按任务自助法 95% 区间 4.7–54.4）。w/o CE 库在 C^orig 下的 89 次成功里 C^faith 只拒绝 2 次。表 7 合计：w/ CE 的 C^orig/C^faith 成功数 165/144，w/o CE 为 89/151。
6. **互换所读判据（§4.2.2，图 5，附录 E 表 8）：** 十个任务上，读 C^orig 的库在 C^orig 下 57.0%、C^faith 下 45.0%；读 C^faith 的库在 C^faith 下 65.0%、C^orig 下 13.5%。表 8：读 C^faith 的库在 C^faith 下 130/200，在 C^orig 下 27/200。
7. **记忆内容（§4.2.3，附录 F）：** 约三分之一的 w/ CE 记忆条目引用判据代码，w/o CE 为零（附录 F 精确数：55 条 w/ CE 中 19 条引用判据代码，53 条 w/o CE 中 0 条）；w/o CE 学习者需要找出物理原因，例如把空夹爪失败归因于 KEEP 指令让半张夹爪合上。
8. **库迁移（§4.3，表 2，附录 G–I）：** 十个源任务蒸馏成一个库，用于互不重叠的十个目标任务（RoboDojo）：空库 65/200，w/o CE 80/200，w/ CE 71/200。LIBERO-PRO（6 任务 × 10 种子）：45/60 对 48/60、46/60；RoboCasa365（10 任务 × 10 种子，库来自 RoboDojo）：22/100 对 24/100、24/100；库来自 RoboCasa365 自身：33/100（w/o CE）、29/100（w/ CE）。跨基准时翻译后的技能几乎不被调用（RoboCasa365 的 200 回合中为 0）。
9. **作者自述局限（§5）：** C^faith 是作者自己写的代理；仅 RoboDojo 改写判据；仿真中的「只看成败」比真实世界的人或 VLM 判成败更干净；只研究了「读 / 不读」两个暴露层级，规则描述、评分表、部分阈值等中间层级没有单独研究；每任务仅 20 个测试布局，故定量结论用汇总，单任务只作定性描述；蒸馏方差未估计；机制解释基于条目文字、技能调用与动作统计，没有消融。

## 对 wiki 的映射

- [paper-rsijudge-criteria-exposure](../../wiki/entities/paper-rsijudge-criteria-exposure.md)
