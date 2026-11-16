# Monkey See, Can Monkey Do? A Benchmark for Evaluating Robot Skill Learning by Observation（arXiv:2609.08209）

> 来源归档（paper）

- **标题：** Monkey See, Can Monkey Do? A Benchmark for Evaluating Robot Skill Learning by Observation（基准名 RoboReel）
- **类型：** paper
- **作者：** Weiwei Gu、Anmol Gupta、Anant Sah（共同一作）、Ryan Varghese、Lalitha Shreya Vanam、Prabhath Adireddi、Peter Karkus（NVIDIA）、Nakul Gopalan（Arizona State University）
- **arXiv：** <https://arxiv.org/abs/2609.08209>（v1，2026-09-08，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2609.08209v1>
- **项目页：** <https://roboreel.github.io>
- **代码：** 论文称数据与基准「在论文录用后发布」（Sec. 1 贡献 1），未给出具体仓库地址
- **入库日期：** 2026-11-12
- **一句话说明：** RoboReel：10 个桌面操作任务的真实人类演示视频（4 视角）加 ManiSkill 3 数字孪生仿真环境与脚本生成的机器人轨迹，4 个测试套件，评测 7 类以上从观察学习（LfO）基线，评测在仿真内完成。

## 核对范围与通读

- **核对范围：** 下载 arXiv v1 PDF 全文并用 pdftotext 逐段阅读：摘要、Sec. 1 引言、Sec. 2 相关工作、Sec. 3 基准构建（3.1–3.4）、Sec. 4 实验（4.1 基线、4.2 结果、4.3 Q1–Q6 分析）、Sec. 5 局限、Sec. 6 结论；表 1–8 与图 1–4 的数据与说明文字；附录 A（PointPolicy 补充结果）、B（视频条件 π0.5 架构）、C（各基线训练细节）、D（数据统计、任务描述与成功判据、仿真配对构建、评测协议）。未读：附录 E 扩展相关工作与参考文献仅扫读；图像内容（如图 5、9、10、11 的实际画面）未逐像素检查，只读了图注；未运行任何代码，未获取项目页与数据集（论文称尚未发布）；OpenReview 在本环境不可访问。
- **通读：** 已完成

## 核心摘录

1. **基准构成（Sec. 3.3 与附录 D.1）：** 实验室桌面环境，18 个刚体物体各有数字孪生；4 路相机（三路外视角 RealSense L515 加一路头盔 D435 自我视角，RGB-D），1280×720、30 fps。每任务每条件 100 条人类演示；ND、WD、PD 各 1000 条，共 3000 条（12000 个 mp4）。
2. **机器人轨迹（Sec. 3.4、附录 D.3）：** 在 ManiSkill 3 / SAPIEN 中用 7-DoF Franka Panda，GraspGen 生成抓取、mplib 规划无碰撞轨迹（五阶段：pre-grasp、grasp、transport、pre-place、drop），为脚本化合成数据而非真机遥操作。ND、WD 各 100 条，共 2000 条。
3. **四个套件（Sec. 3.1）：** ND（无干扰）、WD（训练与评测均有干扰物）、ED（训练无干扰物、评测时引入）、PD（带标定的 3D 手部关键点，仅 PointPolicy 评测）。
4. **评测协议（附录 D.4）：** 每格 3 次评测 × 20 次 rollout = 60 次，报告均值与 SEM；使用预录评测库，仅收录oracle 求解器能成功的初始状态；ND/WD/ED 每任务 80 条人类视频训练、20 条用于评测（提示视频）。
5. **主结果（Sec. 4.2、表 2–4）：** 视频条件 π0.5（均匀抽帧 / BPP 关键帧）整体最强；长时程任务成功率低；PointPolicy 在 Push Cube、Close Drawer 较高，抓取类任务很低（图 4），扩展任务全部 0%（附录 A）。

## 数字与出处

- 表 2（ND）：视频条件 π0.5 关键帧版 Close Drawer 100.0 ± 0.0、Push Cube 100.0 ± 0.0、Press Toaster 88.3 ± 6.7；均匀抽帧版 Close Drawer 98.3 ± 1.7；Stack Cups 两版仅 11.7 / 8.3。SeeDo 在 Bowl in Plate 为 71.3 ± 2.2，高于两个 π0.5 视频条件变体（31.7 / 46.7）。
- 表 4（ED）：关键帧 π0.5 的 Bowl in Plate 25.0、Stack Cups 11.7；SeeDo 的 Bowl in Plate 66.3。
- 图 4（PointPolicy，单任务单模型）：正文称 Push Cube、Close Drawer 表现高且有/无干扰相近，Open Lid、Pick Cube、Bowl in Plate 很低；柱状图数字在 pdftotext 中顺序错乱，本归档不引用具体数值。
- 附录 B：视频条件 π0.5 在公开 openpi/pi05 底座上增加仅两层线性的示范编码器，约 6.5M 参数，示范 token 预算 B = 512（32 帧 × 16 token），语言提示固定为 "complete the task"。
- 附录 C：各基线训练 50k 步（语言条件 π0.5 为 30k 步），单卡 H200，种子 42，取最终检查点。

## 论文内部不一致（核对时发现）

- 图 1 与引言称「over seven hours」「2,500 episodes」；附录 D.1 与表 5 称 3000 条、约 9 小时（表 5 平均 11.56 秒）；摘要/结论称 2,000 条人类视频（即 ND + WD）。三处口径不同，未在论文内说明对应关系。
- Sec. 4.3 Q1 称 keyframe π0.5 在 ED 套件「7 out of 10」任务上优于均匀抽帧；按表 4 逐项比较，严格大于的有 8 项（Close Drawer、Empty Basket、Open Lid、Pick Cube、Press Toaster、Push Cube、Stack Cups、Toys in Basket），Toy in Drawer 持平，Bowl in Plate 反而更低。
- Q1 称视频条件 π0.5 在「所有十个任务、所有套件」最强，但表 2 与表 4 中 SeeDo 在 Bowl in Plate 上更高（见上）。

## 对 wiki 的映射

- [paper-roboreel-lfo-benchmark](../../wiki/entities/paper-roboreel-lfo-benchmark.md)
