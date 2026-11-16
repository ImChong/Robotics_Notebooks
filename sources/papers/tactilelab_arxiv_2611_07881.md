# TactileLab（arXiv:2611.07881）

> 来源归档（paper）

- **标题：** TactileLab: Sim-to-Real Dexterous Robot Manipulation with Efficient Shear-Sensitive Tactile Simulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.07881>
- **PDF：** <https://arxiv.org/pdf/2611.07881v1>
- **项目页：** <https://yijionglin.github.io/tactilelab/>
- **代码：** 未发布（论文称「will be released as an open-source framework」；项目页 Code 入口显示 Coming soon，2026-11-09 抓取核对）
- **版本：** v1（cs.RO，2026-11-05），18 页
- **入库日期：** 2026-11-09
- **一句话说明：** 在 Isaac Lab 刚体仿真里用接触运动学算出稠密「模拟触觉光流」，与接触深度图互补，配 PETS-Net 与 Real-to-Sim-TacFlowNet，在双臂跟随与手内旋转上完成 sim-to-real。
- **核对范围：** 用 pdftotext 通读 v1 PDF 正文第 1–18 页全部章节（摘要、I 引言、II 相关工作、III 仿真器与触觉光流、IV 策略学习与 PETS-Net、V 真实到仿真迁移、VI 实验设置、VII 结果、VIII 讨论与局限、参考文献），表 I–V 与图 5、图 6、图 8 的文字说明逐项核对；图 1–4、7、9–13 只读图注与正文引用，未逐像素查看图像。v1 PDF 全文共 18 页、止于参考文献，不含附录；正文多处引用的 Appendix（观测维度、接触位姿与相对运动范围、训练硬件、附录 C）在该版本中读不到，相关细节记为未核。补充材料视频未看。
- **通读：** 已完成

## 作者与机构（以论文首页为准）

- 作者：Yijiong Lin（1,2）、Wenhao He（2）、Toby Jack（2）、Max Yang（2）、Haoran Li（3）、Nathan F. Lepora（2）。
- 机构 1：OpenDIM Lab，School of Aerospace Engineering，Beijing Institute of Technology（北京理工大学航空航天工程学院）。
- 机构 2：Department of Engineering Mathematics and Bristol Robotics Laboratory，University of Bristol。
- 机构 3：Xi'an Jiaotong-Liverpool University（西交利物浦大学），Suzhou。
- 致谢：ARIA Robot Dexterity 项目（SMRB-PR01-P19）资助；Yijiong Lin 的工作在布里斯托大学完成。

## 核心摘录

1. **模拟触觉光流（§III-C）：** 触觉传感器视为内嵌相机，接触几何由物体表面对名义弹性体表面的几何穿透近似；对接触像素 p，取同一物体表面点在一个仿真步内的刚体相对运动，经射线投影 Π 后的像素位移 F(p) = Π(x_{t+Δt}(p)) − Π(x_t(p))，接触外为 0。不模拟弹性体形变，不需要物体 SDF。
2. **多模态触觉观测（§III-B）：** o_t = {D_t, F_t, c_t}，即接触深度图 D ∈ R^{H×W}、触觉光流 F ∈ R^{H×W×2}、低维接触量（力、位置、剪切）。
3. **PETS-Net（§IV-C）：** 作为 RMA 式两阶段师生框架（沿用 HORA）中学生侧的自适应模块 A_η：对关节角与动作历史做频带位置编码，触觉图经 CNN 空间编码，再用时序编码器聚合，回归教师的特权外参隐向量。
4. **Real-to-Sim-TacFlowNet（§V）：** pix2pix 式生成器，输入连续两帧真实触觉图（通道拼接），输出模拟触觉光流；损失为对抗项加掩码 L1 流重建项。真实侧另有接触深度翻译器，沿用 Tactile Gym 的单帧翻译。
5. **仿真消融（§VII-A，表 I）：** 手内旋转，域内 Time-to-Fail / Rot. / Fall Rate：仅本体 19.87 s / 0.97 / 47.66%；本体+低维接触 25.18 s / 1.21 / 21.33%；本体+深度 26.19 s / 1.21 / 14.26%；本体+光流 25.57 s / 1.24 / 17.45%；本体+深度+光流 27.01 s / 1.28 / 11.76%；Oracle 教师 27.45 s / 1.88 / 9.37%。OOD 物体上深度+光流为 25.08 s / 1.27 / 18.69%，其余各档 Fall Rate 为 29.82%–57.85%。
6. **跨传感器迁移（§VII-C，表 II）：** 6 种传感器配置、FlowMotionNet 仅用仿真数据训练后冻结；无标记 GelSight Mini 的平移 MAE 由直接真实输入的 1.79 mm/s（R² = −0.02）降到经翻译的 0.34 mm/s（R² = 0.98），旋转 MAE 由 4.50 降到 1.18 °/s；无标记 DIGIT 旋转 MAE 仅 3.61 到 3.33 °/s（R² = 0.92）。速度由逐帧位移 MAE 按 25 Hz 采样率换算（表 II 脚注）。
7. **真机双臂跟随（§VII-D1，表 III）：** 仅在仿真用刚性立方体与随机直线轨迹训练；真机 3 个物体 × 直线 / 圆轨迹，位置误差 0.91 mm（刚性立方体直线）到 6.43 mm（泡沫「脑」圆轨迹），姿态误差 0.019–0.057 rad。无对照基线。
8. **真机手内旋转（§VII-D2，表 IV、V）：** LEAP Hand + 四个视觉触觉指尖，仿真中只用胶囊形物体训练；OOD 物体 30 s 上限、每种方法 10 个初始抓取。本体+深度+光流 1.95 ± 0.81 圈、TTT 30.00 s；本体+深度 1.70 ± 0.78；本体+低维接触 1.61 ± 0.90；本体 0.82 ± 0.56。立方体类：深度+光流 1.33 ± 1.00，仅深度 1.04 ± 0.70，低维接触 0.28 ± 0.09，仅本体 0.45 ± 0.35。
9. **作者自述的局限（§VIII-D）：** 迁移假定弹性体与物体处于黏着接触（no-slip），滑移无法在仿真里体现；光流本身可扩展到软体仿真但本文未做；推广到更多物体形状需要更大的高维真实到仿真数据集与几何感知迁移模型。

## 与周报说法的核对

周报来源：具身运控lab 周报（Manipulation 类，第 01 条）。

- 机构：周报写「待核实」。论文首页给出北京理工大学、布里斯托大学、西交利物浦大学（见上）。
- 解决的问题：与摘要 / 引言一致（软体仿真太慢、深度式刚体仿真缺剪切信息）。
- 创新点：「在 Isaac Lab 中用刚体接触运动估计触觉光流，结合接触深度场」与 §III 一致。「真实触觉图像到仿真触觉特征的转换」不够准确：论文翻译的是真实图像对到「模拟触觉光流」与「接触深度图」，不是一般意义的特征。周报未提 PETS-Net，也未提「真实到仿真翻译在无标记传感器上效果参差」这一论文自己的结论。
- 「双臂及手内灵巧操作的 Sim-to-Real 部署」：成立，但双臂任务是 leader / follower 的物体跟随，手内是 LEAP Hand 旋转；仿真里另有 Allegro Hand。
- 周报未出现具体数字，无需回核数字。

## 对 wiki 的映射

- [paper-tactilelab](../../wiki/entities/paper-tactilelab.md)
