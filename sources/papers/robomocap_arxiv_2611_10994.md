# RoboMoCap（arXiv:2611.10994）

> 来源归档（paper）

- **标题：** RoboMoCap: A Large-Scale Whole-Body Motion Capture Dataset with Egocentric Vision for Robot Learning
- **类型：** paper（数据集）
- **arXiv：** <https://arxiv.org/abs/2611.10994>
- **PDF：** <https://arxiv.org/pdf/2611.10994v1>
- **项目页：** <https://bitrobot-foundation.github.io/robomocap/>
- **数据集（HF，非代码仓库）：** <https://huggingface.co/datasets/BitRobot/robomocap>
- **作者与机构：** Mayur Deshmukh、Jiafei Duan（MAGIC Lab，新加坡国立大学）；Aaron Tung、Junyi Li、Michael Cho（BitRobot）；Nikash Bhardwaj、Ademi Adeniji、Vincent Liu（Grounded Superintelligence；部分作者同属 RoboCap 平台团队）
- **入库日期：** 2026-11-11
- **一句话说明：** 把六路头戴相机、双腕相机/IMU 与 120 Hz 53 标记点光学动作捕捉对齐的人体全身数据集，覆盖 291 个动作协议、14 名受试者、200+ 小时，并给出三项基准（第一视角全身重建、足接触与落脚点、向 Unitree G1 的重定向）。
- **核对范围：** arXiv v1 PDF 全文（正文第 1–6 节、表 1–6、图 1–5 的文字与图注，含参考文献列表）；HF 数据集 README 与项目页首页静态读取（未下载数据、未运行任何代码）；附录 / 补充材料：v1 PDF 中未见附录，未另行查找；图 3 流程图、图 4 小提琴图仅读图注与正文数字，未逐像素核对。
- **通读：** 已完成

## 开源状态

- 数据：HF 数据集页，README 标明许可 CC BY 4.0（2026-11-11 静态读取）。
- 代码：HF README 与项目页均写 Code「coming soon」，**未见代码仓库**。

## 核心摘录（均出自论文，章节标注）

1. **规模（摘要、图 1、表 2）：** 200+ 小时、14 名受试者、291 个预定义动作协议、八类（locomotion primitives 44、posture and dynamics 60、loco-manipulation 86、static manipulation 44、long-horizon composite 11、dynamic and recovery 15、human-human loco-manipulation 14、environment-coupled whole-body 17；表中各类 # 与总数 291 的加和以论文表 2 为准）。
2. **硬件（§3.1）：** RoboCap 头戴 6 个全局快门 RGB 相机（1920×1080、30 Hz，硬件同步）+ 2 个 IMU（200 Hz）+ 磁力计（100 Hz）；左右腕各一个 RoboWrist；1 个外部第三人称相机；NOKOV 光学动捕，53 个反光标记，120 Hz；头戴上 4 个标记构成 Tracker0；物体用 4 个标记做 6-DoF 跟踪；麦克风录音。
3. **环境（§3.2）：** 在 NOKOV 动捕棚 7×7 m 区域采集，布置窄道、障碍、斜面、高度变化。
4. **同步与标定（§4.3）：** 以头戴 IMU 陀螺角速度幅值与 Tracker0 角速度做互相关估时偏，20 秒窗口取中位数，漂移显著（>3 ms）才用仿射时间映射；各模态保留原采样率，不重采样。
5. **标注（§4.4）：** 对光学数据拟合 SMPL-X；手部为视觉估计（Grounded API，MANO），论文明确与光学身体测量区分；场景用 MapAnything 重建；任务文本与起止时间戳。手部模型在 34.9 分钟训练集上微调、9.2 分钟测试集上报 PA-MPJPE。
6. **质量（§4.4–4.5）：** 对齐中位峰值相关 0.985；窗口偏移相对整段估计中位偏差 0.5 ms（95 分位 26 ms）；拟合脚底中位高出地面 10 mm；99.81% 有效骨架帧、99.74% 标记可见率、SMPL-X 标记重建误差中位 4.78 mm；三类核心动作合计 161 小时（约 80.6%）。
7. **多样性（§4.5.2）：** 前进行走中位头朝向—行进方向偏差 9.5 度，侧向/后退行走中位 92.4 度（图 4C）；步宽中位前进 14.4、转向 13.6、侧/后 15.7、携带 14.1 cm。
8. **基准一（§5.1，表 3）：** 公开预训练的 AGRoL、EgoPoser、EgoEgo、EgoAllo，不在 RoboMoCap 上训练或微调；EgoAllo 的 MPJPE 最低 80.8 mm，AGRoL 的 W-MPJPE 最低 95.4 mm。
9. **基准二（§5.2，表 6）：** EgoAllo 接触 F1 90.5%、落脚点误差 85.1 mm、步位移误差 154.1 mm、滑步 54.8 mm/s；GT SMPL-X 对声明标记标签的接触 F1 也只有 93.6%。
10. **基准三（§5.3，表 4–5）：** 543 个测试 session（约 18 小时，30 Hz）重定向到 29-DoF Unitree G1，对比 GMR、PHC、OmniRetarget：OmniRetarget 全局关键体位置误差 34.8 mm（GMR 90.7、PHC 173.9），接触 F1 87.7%，滑步率 2.2%；但片段成功率 OmniRetarget 53.6%、PHC 79.6%、GMR 7.4%；三法超过 70% 帧接近关节限位。
11. **论文自述局限（第 6 节）：** 仪器化光学动捕环境；偶发标记遮挡与身体模型拟合误差；受试者/物体/场景有限；人与人形形态差异需重定向与控制适配；下游人形执行验证留作未来工作。

## 对 wiki 的映射

- [paper-robomocap](../../wiki/entities/paper-robomocap.md)
