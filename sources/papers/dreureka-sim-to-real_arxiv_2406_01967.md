# DrEureka: Language Model Guided Sim-To-Real Transfer（arXiv:2406.01967）

> 来源归档（paper）

- **标题：** DrEureka: Language Model Guided Sim-To-Real Transfer
- **作者：** Yecheng Jason Ma*、William Liang*、Hung-Ju Wang、Sam Wang、Yuke Zhu、Linxi "Jim" Fan、Osbert Bastani、Dinesh Jayaraman（宾夕法尼亚大学；NVIDIA；UT Austin）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2406.01967>（v1，2024-06-04，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2406.01967>
- **项目页：** <https://eureka-research.github.io/dr-eureka>
- **代码：** https://github.com/eureka-research/DrEureka
- **入库日期：** 2026-11-08
- **一句话说明：** 用 GPT-4 先生成带安全指令的奖励函数（基于 Eureka），再用「奖励感知物理先验（RAPP）」约束 LLM 生成域随机化配置，只需目标任务的物理仿真即可完成零样本 sim-to-real；在 Go1 四足行走、LEAP Hand 方块旋转上对标人工设计，并演示 Go1 在瑜伽球上行走。
- **核对范围：** 下载 arXiv PDF v1（pdftotext 全文，逐段通读）：摘要、I 引言、II 相关工作、III 问题设定、IV 方法（Eureka、安全指令、RAPP、DR 生成）、V 实验设置、VI 结果（人工设计对比、DR 消融、Walking Globe）、VII 结论、VIII 局限，以及附录 A（提示词与成功判据）、B（LLM 生成的奖励与 DR 输出）、C（奖励数学表示，略读）、D（三个任务的实验设置与 DR 参数表）、E（附加消融）、F（CEM/BayRn 基线）；图 2、4、5 对照渲染页核对。未做：未运行官方代码；项目页视频未看；参考文献所引原文未逐篇核对；GitHub 仓库页在本环境返回 403，未能打开，开源状态依据项目页上的链接。
- **通读：** 已完成

## 开源状态

- **已开源**：论文首页给出项目页，项目页链接 <https://github.com/eureka-research/DrEureka>（本环境直接访问仓库返回 403，未核对仓库内容；步骤 2.5 以项目页链接为准，2026-11-08）。

## 核心摘录

1. **问题与动机（摘要，I）：** sim-to-real 通常依赖人工设计奖励与调仿真物理参数，慢且费人力；DrEureka 只要目标任务的物理仿真，就自动构造奖励与域随机化分布。
2. **三阶段分解（I，IV）：** 同时联合搜索奖励与 DR 参数是「无限维函数空间」搜索，对 LLM 昂贵低效，因此拆成：(a) Eureka 生成带安全指令的奖励（式 5、6，Algorithm 1）；(b) 用初始策略在被扰动的仿真里测可行范围，得到 RAPP（Algorithm 2）；(c) 把 RAPP 范围给 LLM，让它选参数子集并定范围，零样本生成多个 DR 配置，各训一个策略（式 7）。
3. **RAPP（IV-C）：** 对每个 DR 参数，在一组通用候选值（四类通用范围：0_to_infty、0_to_1、centered_0、centered_1，附录 D1 表 XII 注）上逐一设置、其余保持默认，滚动初始策略；满足预设成功判据的取值视为可行，上下界取可行值最小最大。作者假设并声称验证：仿真内置范围过宽会妨碍策略学习。
4. **主要实验设置（V）：** Go1 四足前进 2 m/s（实验室 5 米轨道），仿真与控制器沿用 Margolis 等人；LEAP Hand 方块旋转（20 秒内尽量多转）；骨干 LLM 为 GPT-4；每任务采样 16 个 DR 配置，全部训练并上真机评估，报告最佳与平均；PPO；仿真为 Isaac Gym；每个 DR 配置 3 个随机种子（V「Policy Training and Evaluation」）。
5. **前进行走结果（表 I，VI-A）：** Human-Designed 1.32±0.44 m/s、4.17±1.57 m；Eureka（无安全指令、无 DR）0.0 与 0.00（真机不能走）；DrEureka 最佳 1.83±0.07 m/s、5.00±0.00 m，平均 1.66±0.25 m/s、4.64±0.78 m。引言称在「各种真实评测地形」上前进速度高 34%、距离高 20%。
6. **方块旋转（表 II）：** Human-Designed 3.24±1.66 rad、跌落时间 20.00 s；DrEureka 最佳 9.39±4.15 rad、20.00 s；平均 4.67±3.55 rad、16.29±6.28 s。
7. **真实世界鲁棒性（VI-A，图 4、图 5）：** 在 Default（实验室默认）、Grass（实验室内人造草皮）、Sidewalk（室外空人行道）、Socks（给机器狗腿套袜子）四种条件下比较 DrEureka（Best）与 Human-Designed，作者称前者「持平或优于」后者；图 5 为柱状图，正文无逐项数字。
8. **消融（表 I、表 XV、表 XVI）：** 无 DR 1.21 m/s；人工 DR 1.35；Prompt DR（直接用 RAPP 全范围）1.43；无先验 0.09、无信息先验 0.08（每组 16 个策略中 15 个动作剧烈危险，记 0）；随机采样 0.98；CEM Random 0.00；CEM RAPP 1.46；BayRn RAPP 1.28。安全指令消融：不含安全指令时仿真 1.83 m/s 但真机 0.0（面朝下摔倒），含安全指令 1.70 仿真 / 1.21 真机（均为无 DR 变体）。注意表 I 中「人工 DR」的距离 4.83 m 高于 DrEureka 平均 4.64 m。
9. **效率（VI-B）：** DrEureka 墙钟约 3 小时，对比 CEM 10 小时、BayRn 20 小时；作者指出 CEM、BayRn 需反复在真机评估中间策略，DrEureka 无需。
10. **Walking Globe（VI-C，表 III）：** Go1 在 34 英寸瑜伽球上；仿真 10.7±5.20 s，实验室真机（有人作中心支撑 / 牵引）15.4±4.17 s；室外多种地面累计运行 4 分钟以上（草地、人行道、木桥等，限速、人牵引）。仿真使用 Ji 等 Dribblebot 的执行器网络（附录 D3）。
11. **局限（VIII）：** 无视觉输入；DR 参数生成后训练中固定；没有从仿真表现预测真机表现的策略选择机制。

## 作者自述 vs 本页判断

- 自述：DrEureka 是首个研究 LLM 自动合成 DR 配置、并把自动奖励与 DR 结合用于敏捷技能的工作（II）。
- 判断（推测）：对比对象只有每任务一个人工配置，且人工奖励依赖速度课程（V），结论更像「可自动得到可用配置」而非「普遍优于人类」；作者自己也写绝对排序不重要（V）。

## 对 wiki 的映射

- [paper-dreureka-sim-to-real](../../wiki/entities/paper-dreureka-sim-to-real.md)
- [domain-randomization](../../wiki/concepts/domain-randomization.md)
- [sim2real](../../wiki/concepts/sim2real.md)
