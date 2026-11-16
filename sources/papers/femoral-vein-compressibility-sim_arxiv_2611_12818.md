# Simulating Vein Compressibility for Ultrasound-Guided Femoral Access（arXiv:2611.12818）

> 来源归档（paper）

- **标题：** Simulating Vein Compressibility for Ultrasound-Guided Femoral Access: Speed, Accuracy, and Phantom Validation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.12818>
- **PDF：** <https://arxiv.org/pdf/2611.12818v1>（v1，2026-11-09）
- **代码：** https://github.com/MRDLab/femoral-sim-calibration
- **作者 / 机构：** A. Manicka、L. Coulon、D. Ang、M. Loushin、T. M. Kowalewski；明尼苏达大学（机械工程、麻醉科 Visible Heart Lab）、Lightside Surgical 等（首页作者栏）
- **版面：** 首页注明 Journal of Medical Robotics Research 版式；仓库 README 称其为 ISMR 2027 投稿。两处说法不同，以 PDF 首页与仓库 README 各自所写为准，此处不做裁定。
- **入库日期：** 2026-11-11
- **一句话说明：** 在 Isaac Sim for Healthcare 中用 Stable Neo-Hookean 四面体网格加 Vertex Block Descent（VBD）单 GPU 求解，模拟探头按压股部血管，对比低保真实体仿体并扫描求解器参数的速度与精度。

## 核对范围与通读

- **核对范围：** 读了 arXiv v1 PDF 正文全部 11 页（摘要、第 1 至 5 节、表 1 至 3、图 1 至 4 图注、参考文献）；读了仓库 `supplement.pdf` 全文（S1 至 S3、表 S1 至 S9、图 S1 至 S7 的图注与表）；读了仓库根 README。未读：仓库 `i4h-phantom/` 源码；未运行任何代码；图为只读图注与对应表数值，未逐点读图像本身；补充材料以仓库 main 分支版本为准，未核对其与投稿版是否一致。
- **通读：** 已完成

## 开源状态

- 仓库 README（静态读取）：含论文 `main.pdf`、`supplement.pdf`、视频 `clips/`、仿真代码 `i4h-phantom/`；README 称不含 LaTeX 源、仿体实测记录（表 S5 给出全部测量值）与约 5 GB 原始运行输出。需另行克隆 NVIDIA Isaac for Healthcare、Isaac Sim、Isaac Lab 并构建一个打补丁的超声容器。

## 核心摘录（均出自论文正文）

1. **设定（§1、§2.2）：** 探头压力使组织变形、改变机器人看到的血管；以「静脉先于动脉塌陷」为临床信号（压迫超声区分静脉与动脉）。组织为四面体有限元网格，Stable Neo-Hookean 本构，VBD 求解，探头与血管壁接触用罚能量；渲染用 Isaac for Healthcare 的 raysim 光线追踪 B 超。
2. **硬件与软件（§2.2）：** 单工作站 RTX 5090（32 GB）、Intel Core Ultra 9 285、64 GB 内存；Newton 1.2.1 / Warp 1.13.0 / CUDA 12.9；Isaac Sim 6.0.1。一帧定义为 16.7 ms 仿真时间（60 Hz）。
3. **仿体验证（§2.4、§3.1）：** 圆柱形低保真仿体（肌肉核 + 脂肪层，硅胶管充当血管，动脉 9 mm 孔径 70A、静脉 10 mm 孔径 35A），UR5 机器人按压到 7 个深度（最大 25.4 mm）各 3 次。动脉在全部 7 个深度与仿真相差 5 个百分点以内（其中 6 个深度在 3 点内）；静脉到 11 mm 在 3 点内，15 至 25.4 mm 仿真静脉比实体塌得更多，差 5.3 至 11.9 个百分点；以首次接触高度为静息直径时静脉差异最大 16.4 个百分点。表 S5：静脉在 7 个深度中只有 3 个在 5 点内（误差 15 mm −6.1、19 mm −7.6、23 mm −11.9、25.4 mm −5.3），动脉 25.4 mm 为 +4.6；**两个仿体都没有区分开静脉和动脉**（相差不超过 8 个百分点）。「未对仿体调参」（§2.4 末句 No input was adjusted to the phantom）。
4. **求解器扫描（§3.2、§3.3、表 3）：** 切片（slab）上 1.5 mm 单元最终选定但更细网格仍改变结果（25.4 mm 处静脉余量 1.5 mm 为 33.8%、1.0 mm 为 19.5%）；腿模型选定 8.33 ms 时间步（每帧 2 步）、每步 35 次迭代；墙钟时间与求解工作量线性相关（每次求解遍历 0.400 ± 0.001 ms），精度不随工作量提高；12 个设置均在 3 点容差内，最大差 1.20%。CUDA Graph 在同设置下约 3 倍加速（表 S3）。
5. **速度（摘要、§3.3、§4.3，表 S3、S4）：** 选定设置仅求解器 30.4 ms 每帧；摘要的 29,700 ms 是参考设置**不用 CUDA Graph** 的墙钟，正文写 977 倍，摘要写 976 倍。同为 CUDA Graph 开启时，参考设置为 9,302 ms，选定设置的帧加速为 306 倍（表 S4），整次按压加速 203 倍；含超声与 3D 渲染的完整回路 56.4 ms 每帧（一次运行），慢于 16.7 ms 实时线（补充视频段：整段运行约 0.29 倍实时，表 S9），但在文中引用的扩散策略控制器的 43 至 100 ms 动作间隔内。
6. **腿模型临床信号（§3.4）：** 用文献参数（表 1）时，11 mm 处静脉保留静息面积 75.4%、动脉 99.7%；25.4 mm 处 27.7 ± 0.1% 对 83.7 ± 0.1%，相差 56 点；所有 12 个求解设置、5 次运行都如此。静脉读数受接触间距形成的「闭合下限」（名义 14 至 20%）限制。
7. **局限（§4.4，作者自述）：** 血管无血压、血流、搏动；休克与心脏骤停下表现未知；无已发表的刚度是在此类加载下测得；腿模型的脂肪剪切模量约为在体大腿中位数的 2 倍、肌肉约 4 倍；静脉壁 20 kPa 对文献拉伸测试 5,700 kPa 差约 285 倍；不报告探头力；仿体对比仅一个仿体、一个层面，模量为供应商值，面积由单个内高度推算；声学材料用 NVIDIA 的肝脏工作流默认值，模拟图像未与临床图像比较。
8. **与文章说法相关的原文限定：** 摘要末句「suggests that the NVIDIA Isaac Sim platform used may be suitable to augment training for reinforcement learning or online simulation for autonomous femoral access」，是可行性判断，论文中没有强化学习训练实验。

## 对 wiki 的映射

- [paper-femoral-vein-compressibility-sim](../../wiki/entities/paper-femoral-vein-compressibility-sim.md)
