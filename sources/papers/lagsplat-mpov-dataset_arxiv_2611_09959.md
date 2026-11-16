# A four-view smartphone video dataset of a robot-held flexible strip（arXiv:2611.09959）

> 来源归档（paper）

- **标题：** A four-view smartphone video dataset of a robot-held flexible strip, with camera poses and a 3D LaGSplat baseline
- **类型：** paper（数据集说明 note，PDF 共 2 页）
- **作者 / 机构：** Louen Pottier；Université Paris-Saclay, CEA, List（法国 Palaiseau，首页署名）
- **arXiv：** <https://arxiv.org/abs/2611.09959>（v1，cs.CV，2026-10-26）
- **PDF：** <https://arxiv.org/pdf/2611.09959v1>
- **代码：** https://github.com/LouenPottier/lagsplat-mpov
- **项目页：** <https://louenpottier.github.io/xrdays.html>（论文称其为浏览器 3D 演示；本次未打开，未核实）
- **数据：** Zenodo 10.5281/zenodo.23601189，CC BY-NC 4.0（论文 Availability 段）
- **入库日期：** 2026-11-10
- **一句话说明：** 同步四路 iPhone 视频、实测相机位姿与机械臂一维输入 s(t) 的柔性条带数据集，并给出 LaGSplat 的 3D 基线。
- **核对范围：** 下载 arXiv v1 PDF 全文（2 页，含正文、Table 1、Limitations、Availability、参考文献），pdftotext 逐段通读；图 1 仅读到图注，未逐像素核对图像内容；未下载 Zenodo 数据与检查点；仓库 README 前部经 raw 文件静态阅读（未运行代码）；被引的 LaGSplat 原文（arXiv:2608.16324）未读，其内容均不在此归档中断言。
- **通读：** 已完成

## 开源状态

- 数据集：Zenodo（CC BY-NC 4.0，论文 Availability 段）；仓库 README 另称 Hugging Face 有镜像（README，静态读，未核实内容）。
- 训练代码：仓库 README 称为“产出已发表模型的研究代码，路径已适配数据集”，重新训练“得到相近而非完全相同的数值”（README，非论文）。

## 核心摘录（均出自论文正文，段落名为论文自带小标题）

1. **采集设置（Setup）：** 机械臂夹持约 23 cm 的柔性条带，末端沿近一维水平路径“猛拉”（jerk），之后条带自由振荡并衰减。四部 iPhone 以 1920×1080、60000/1001 fps 拍摄；iphone0/1/2 固定且相互刚性固连（iphone0 机型与其余不同），iphone3 手持，前 7 分钟绕场景移动，之后近乎静止。另在物体全静止的独立会话中用 iphone3 拍两段房间静态片段。
2. **同步（Synchronization）：** 粗偏移来自拍手声与机械臂运动（猛拉间隔不规则，故无歧义）；各机时钟率不等，iphone0 全程相对其余三台漂移 9 帧。四路视频统一重采样到 iphone1 时钟，取最近源帧，残差至多半帧（8 ms）；每路 36 734 帧（612.8 s）。
3. **相机几何（Camera geometry）：** 未拍摄标定板，位姿全由场景求得；世界系取 iphone0 相机系（OpenCV 约定），长度单位为 iphone0 与 iphone1 间距（约 0.38 m）。固定相机：对手工点击的对应点做联合光束法平差解焦距、径向畸变与位姿，检验为仅用 iphone0/1 三角化再投影到 iphone2，偏差 3 至 6 px。手持相机：对 iphone3 自身帧建 SIFT 地图后逐帧 PnP，剔除内点过少或非物理跳变的位姿，36 734 帧中 25 381 帧有位姿。静态片段：关键帧上 COLMAP，相似变换对齐到世界系；有一块位姿曾被放到近对称柜体的错误一侧，已纠正并标记。
4. **状态与区域：** state.csv 给出每帧 s(t)（夹爪沿路径弧长，在 iphone1 中跟踪，作为系统实测输入）及基线编码出的条带潜状态 q(t)∈R²（仅供参考）；zones.json 给出世界系下两个有向盒：zone q 含条带，zone s 含手臂运动部分。
5. **基线（Baseline）：** 在 [1]（LaGSplat）基础上三处改动：(i) 机械臂作为输入，解码器以 (s, q) 为条件，动力学为 M q̈ = −∇V(q) − C q̇ + G s̈（M、C 为常数，V 为单极小的 invex 势）；(ii) 仿射解码器换成共享形变网络，输出每个高斯的位移、尺度、旋转与颜色，J 不再恒定而在当前状态计算；(iii) 背景为静态片段重建的房间，仅 zone q 内高斯依赖 (s, q)，zone s 内仅依赖 s。模型约 10^5 个高斯（PDF 文字层为“105”，推测为上标丢失的 10^5，需对照原 PDF 排版），相机位姿训练中细化。
6. **基线结果：** 末 60 s 留出。渲染图（960 px）PSNR：固定相机 31.4 至 32.8 dB，手持 26.5 dB；限定在 zone q 投影内为 28.4 至 30.0 dB（固定）与 26.2 dB（手持）；静态场景留出图 24.7 dB（逐颜色通道做增益与偏置后）。动力学：自实测状态起 rollout，RMS 误差除以 q 的标准差，0.25 s 为 0.53，2 s 为 0.83。
7. **论文自述局限（Limitations 段）：** 标定误差为几像素而非亚像素；handheld 段部分 iphone3 位姿错误；s(t) 仅在一个视角中以像素读取；iphone0 的颜色与其余相机不同。

## 对 wiki 的映射

- [paper-lagsplat-mpov-dataset](../../wiki/entities/paper-lagsplat-mpov-dataset.md)
