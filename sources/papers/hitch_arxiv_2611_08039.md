# HITCH（arXiv:2611.08039）

> 来源归档（paper）

- **标题：** HITCH: Teaching a Vision-Language Model to Operate a Humanoid through a Whole-Body Harness
- **类型：** paper
- **作者：** Qingzhou Lu, Chenxin Liu, Xinming Wang, Liu Cao, Yanjiang Guo, Mengdi Xu, Jianyu Chen（等贡献标注见论文首页脚注）
- **机构（论文首页脚注）：** 1 Tsinghua University；2 Institute of Automation, Chinese Academy of Sciences；3 Shanghai Qizhi Institute（脚注未在 HTML 文本中标出每位作者对应编号）
- **arXiv：** <https://arxiv.org/abs/2611.08039>（v1，cs.RO，2026-11-05）
- **HTML 全文：** <https://arxiv.org/html/2611.08039v1>
- **PDF：** <https://arxiv.org/pdf/2611.08039>
- **项目页：** <https://hitch.axell.top>（前端渲染页，静态抓取只得到空壳，未能确认是否有代码 / 数据链接）
- **入库日期：** 2026-11-09
- **一句话说明：** 给 VLM 一套「感知（选图像点返回 3D 位置）+ 执行（按身体部位下命令，由规划器和学习型跟踪器变成全身动作）」的 harness，用仿真教师代码策略在五个基础任务上的轨迹微调 VLM，不用目标基准的任务轨迹即可在 SIMPLE / HumanoidArena 取得一定成功率，并在真实 G1 上做搬箱与导航。

## 核对范围与通读

- **核对范围：** 读 arXiv HTML v1 全文：摘要、I 引言、II 相关工作、III 方法（III-A 至 III-E）、IV 实验（IV-A 至 IV-G，Table I 至 IV）、V 结论与局限，附录 A 接口与示例对话、B 基础任务、C 训练超参、D 执行实验细节、E 指令对照（Table V 至 VI）、F 时限与时长（Table VII）；参考文献列表只核对了与本页相关的条目（[10] RAX、[11] HumanoidArena、[18] HEFT、[19] SONIC、[4] SIMPLE）。图（Fig. 1 至 6）只读了图注与正文引用，未看图像本身。未读项目页内容（静态抓取为空壳）、未下载或运行任何代码。注：直接下载 arxiv.org/pdf/2611.08039 时本次得到的文件内容是另一篇论文（H2 MARL），故本归档全部依据 HTML v1。
- **通读：** 已完成

## 核心摘录

1. **harness 两个接口（III-B、III-C）：** 执行接口一条命令按固定顺序设定 body（骨盆高 0.30 至 0.79 m，躯干俯仰 / 偏航）、hands（每手 1 至 12 个手掌路点）、fingers、walking、gaze；规划器做逆运动学并生成步态后，以 50 Hz 的跟踪器执行。感知接口让 VLM 在最新图像里选 1 至 100 个点（坐标归一化到 0 至 1000），按深度反投影返回观察系 O 下的 3D 位置（附录 A）。
2. **跟踪器（III-B2、Table I）：** 基于 HEFT，再用规划器在基础任务里产生的参考继续训练，并对躯干和手加随机外力。Table I：手部误差 SIMPLE 0.6 cm / HumanoidArena 0.7 cm，步行误差 5% / 3%；教师在 240 个 episode 上成功率 97.9%，对照 HEFT 94.2%、SONIC 27.9%、AMO 12.9%。
3. **教师与训练数据（III-D、III-E、附录 B、C）：** 五个基础任务（单手抓放、双手搬箱、绕障导航、开关抽屉柜门、推拉房门）各 500 条轨迹；教师是仿真中的代码策略，只许用与学生相同的观察，论文说由 Claude Opus 5.5 编写；只保留成功 episode。微调 Qwen3.8-27B（原文名称）的全部语言模型权重，视觉编码器冻结，仅对教师回复计 loss；学习率 5e-6，全局 batch 16 条轨迹。
4. **零任务轨迹迁移（IV-C、Table II）：** 仅基础任务训练，SIMPLE 成功率 3.3%（预训练）升到 51.1%（6 个任务，每个 30 个评测 episode，共 180 个）；HumanoidArena 4 个任务子集（导航、拳击、开门、搬箱上架，各 20 个 episode，共 80 个）16.3% 升到 42.5%。Qwen3.8-Max（API，未微调）SIMPLE 为 12.2%（不思考）/ 7.8%（思考）（Table III）。
5. **少量轨迹（IV-D、Table II）：** 每任务 5 条 SIMPLE 轨迹，单独训练 90.0%，加基础任务 95.0%；10 条为 95.0% 对 98.9%；30 条 96.7% 对 98.9%。但基础任务加 10 条 SIMPLE 轨迹后 HumanoidArena 从 42.5% 降到 28.8%。
6. **去掉感知（IV-E）：** 基础任务模型 SIMPLE 从 51.1% 降到 2.8%，HumanoidArena 从 42.5% 降到 27.5%。
7. **推理（IV-F、Table III）：** 微调后带推理的模型在 SIMPLE 上更差（如基础任务 + 10 条：98.9% 对 89.4%）；HumanoidArena 上无一致优势。
8. **真机（IV-G、Table IV）：** 仅基础任务模型、不带推理、训练无真实图像；搬箱 13/20（朝桌 8/10，朝折叠桌 5/10），导航 6/10；成功 episode 平均搬箱 60.7 s、导航 26.8 s，其中执行占 61%。真机手为橡胶手，无手指控制。
9. **评测口径（IV-A、附录 E、F）：** 两个基准都换成论文自己的头部相机（2 自由度云台 + RealSense D455），且重写了指令（Table VI）；成功判据时限 SIMPLE 45 s、HumanoidArena 60 s，长于官方时限（Table VII，如 SIMPLE BendPick 官方 8 s）。
10. **论文自述局限（V）：** 规划器不支持坐下、跨障碍等动作；加入 SIMPLE 轨迹会降低 HumanoidArena 成功率；微调后推理在 SIMPLE 反而降低成功率；命令执行期间无法反应；真机手无手指控制只能做两个任务。

## 周报说法核对（两份周报）

- **eiBI 周报：** 机构（清华大学、中科院自动化所、上海期智研究院）与论文首页脚注一致。「由目标点、骨盆、躯干、双手、行走和视线命令组成的全身接口」：论文里「点」属感知接口（选图像点取 3D 位置），不是执行命令；执行命令是骨盆 / 躯干 / 双手（及手指）/ 行走 / 视线。「教师策略轨迹」属实（代码策略，非人类专家）。「长时移动操作」一词论文未用；成功 episode 平均耗时约 12 至 34 s（附录 F，Table VII），论文未宣称长时程。
- **Y21u 周报：** 机构待核实，补全同上。「专家轨迹」更准确为仿真里的教师代码策略轨迹。51% / 42.5% 对应 Table II 的 51.1%（SIMPLE 6 任务）与 42.5%（HumanoidArena 4 任务子集，非全部任务），属实但需说明范围和取整。「在真实 G1 上完成搬箱行走」：真机成功率是搬箱 13/20、导航 6/10，并非稳定完成。

## 开源状态

- 论文正文未给出代码或数据链接；项目页 <https://hitch.axell.top> 静态抓取为空壳，开源状态**未确认**（步骤 2.5 未能完成核查）。

## 对 wiki 的映射

- [paper-hitch](../../wiki/entities/paper-hitch.md)
