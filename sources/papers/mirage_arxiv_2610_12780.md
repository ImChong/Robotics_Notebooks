# MIRAGE 论文来源归档

## 元数据

- **题目：** Multimodal Image-based Robotic Assistance and Guidance Environment (MIRAGE): A Novel Unified Framework for Surgical Navigation and Visualization
- **作者：** Ehsan Nasiri, Yi Zheng, Michael A. Kokko, Xiaoyao Fan, William R. Warner, Keith D. Paulsen, Ryan J. Halter
- **单位：** Thayer School of Engineering, Dartmouth（Hanover, NH, USA）
- **预印本：** https://arxiv.org/abs/2610.12780 （v1，2026-10-08，cs.RO；投稿 ASME Journal of Mechanisms and Robotics，审稿中）
- **项目页：** https://sites.google.com/dartmouth.edu/halter-lab/resources/mirage （只有简介与结果摘要，无下载）
- **经费：** ARPA-H award 1AY2AX000043；作者声明已提交美国临时专利申请 63/920,165 与 64/166,677
- **代码：** 未公开（论文声明实现含专有硬件接口，原始采集日志与完整实现不公开）

## 论文要解决的问题

达芬奇术中需要在外科医生控制台里同时看到 3D 重建、AR 叠加、导航数值和术中超声，各模态又来自不同设备、时钟和坐标系。论文提出一个时间同步、可量化评估的统一框架，并做临床前模体验证。

## 方法与训练信息

- **架构：** C++ 后端（Bluefish 采集左右 RGBA 帧、光学/电磁追踪接口、OpenGL 渲染、AR 叠加、超声与标定管线）+ Python 前端（GUI、配准、HUD 导航、立体重建、模块启动器），经 pybind11 的 OverlayEngine 绑定，带时间戳帧缓冲、互斥保护状态和工作线程，声称可能时零拷贝。
- **时间：** 全部流用单调主机时钟；追踪位姿对视频时间戳做位置线性插值、旋转 SLERP。
- **配准：** Polaris Vega XT（光学）与 Aurora（电磁）；CT 基准点与探针采样点之间最小二乘 SVD 刚性拟合并显示 RMSE；实时导航线程组合工具位姿与 CT 配准、平滑并投影回 SO(3)。
- **立体标定：** OpenCV 标定内外参，自制棋盘格/反光球装置，外参用 SE(3) 扭量参数化。
- **立体重建：** 视差由 RAFT-Stereo 或 PSMNet 给出，Q 矩阵反投影，双边滤波、视差分位截断、IQR 深度门限、离群点移除；VTK 渲染并与半透明 CT 网格对齐。
- **超声：** BK Medical 3000，DICOM 文件或 OpenIGTLink 实时流；有界队列满则丢最旧帧；多边形 ROI、标记点导出 CSV、marching cubes 伪体积；探针追踪时 US→CT 变换由三段变换相乘。
- **HUD：** 在 TilePro 显示工具到目标距离与接近角，按归一化距离-角度得分选目标；3D 场景与 HUD 最高 40 Hz。
- **硬件：** Kronos K8 SDI 采集卡 + RTX 6000 Ada GPU 的 PC。

## 论文报告的结果

- 平台：da Vinci S、Si、Xi、SP；模体：咽喉、肾脏、手术模体与含毛细管的杆模体；工作距离 100/300/500 mm（Xi、SP）。
- 摘要：平均 3D 重投影误差 0.6 mm，最佳 RMSE 0.1 mm（Xi，100 mm）。表 2：100/300/500 mm 物理误差 0.1/0.3/0.6 mm，对应像素误差 2/2.5/2.5 px。**摘要的 0.6 mm 均值与表中三档平均约 0.33 mm 不一致，0.6 mm 只对应 500 mm；论文未解释。**
- 标定重投影 RMSE：S 约 6 px、Xi 约 2 px（正文）；表 5 中 Xi 优化后 2D 误差 4.3→2.5 px、3D 误差 3.0→0.6 mm（两处 Xi 像素数不完全一致）。
- 基准点配准 RMSE（表 4，均值±SD）：S/Si 1.2±0.20 mm，Xi 0.9±0.20 mm，SP 0.9±0.20 mm；未报告 TRE 数值。
- AR 叠加误差（100 mm，表 7）：S 6.0、Si 3.0、Xi 2.5、SP 2.5 px。
- 立体重建误差（表 6，mm；PSMNet/RAFT-Stereo）：腹主动脉 2.63/2.17，下腔静脉 2.30/1.79，右肾 2.15/2.25，右输尿管 2.43/2.13；单对推理约 0.60 s / 0.20 s。
- 时序：叠加配对帧显示延迟 16.7 ms（60 FPS，追踪 400 Hz）；HUD/超声追踪 40 Hz；端到端延迟未给数值（表 8 仅为报告结构）。
- 超声无定量精度或延迟。

## 复现与解释边界

- 全部为模体实验，不代表临床有效或优于现有流程；人因、学习曲线、临床结局均未评估。
- 贡献主要是现有方法的集成、评估和跨语言绑定；SP 平台实验最少。
- 实现不公开，复现依赖论文的数学形式、伪代码、配置模式与设备无关评测协议。
- 附录与第 6 节部分文字抓取未读到；已通过第二次分段抓取读到实验、讨论与可用性声明。

## 一手来源

- <https://arxiv.org/abs/2610.12780>
- <https://arxiv.org/html/2610.12780v1>
- <https://sites.google.com/dartmouth.edu/halter-lab/resources/mirage>
