# MidasTouch & YCB-Slide（原文与仓库核对笔记）

- **来源：** <https://arxiv.org/abs/2210.14210>（`arXiv:2210.14210v1`，CoRL 2022，2022-10-25）
- **项目页：** <https://suddhu.github.io/midastouch-tactile>
- **数据集与采集代码仓库：** <https://github.com/rpl-cmu/YCB-Slide>（MIT License）
- **作者与机构：** Sudharshan Suresh, Zilin Si, Stuart Anderson, Michael Kaess, Mustafa Mukadam（CMU Robotics Institute & Meta AI Research）
- **核对日期：** 2026-11-14
- **核对范围：** MidasTouch 论文全文 `arXiv:2210.14210v1`（第 1–755 及 801–1312 行，不含参考文献末尾与附录标题共约 45 行）及 `rpl-cmu/YCB-Slide` 仓库 `README.md`、`LICENSE`、`download_dataset.sh`、`ycb_slide/record/*.py`、`ycb_slide/viz/*.py`
- **通读：** 已完成

---

## 开源与数据核查

- **开源状态：** **数据集与采集/可视化工具已开源**。`rpl-cmu/YCB-Slide` 以 MIT License 开源了完整数据集下载脚本（`download_dataset.sh`，经 Google Drive 获取 sim、real、DIGIT 背景与 YCB 真值网格）、实时采集同步管线及轨迹/触觉可视化脚本；需注意 `README.md` 文档中的脚本命令示例名称（如 `record_data_expt.py`、`timesync_digit.py`）与仓库包内实际脚本文件名（`ycb_slide/record/record_data.py`、`ycb_slide/record/timesync_data.py`）存在轻微命名差异，运行时应以包内实际文件为准。

---

## 原文核心事实摘录

1. **YCB-Slide 数据集定位与规模：** 针对未知全局初值的**在物滑动触觉定位（tactile localization from sliding touch）**基准，选取 **10 个几何形态各异的 YCB 物体**（`sugar_box`、`tomato_soup_can`、`mustard_bottle`、`bleach_cleanser`、`mug`、`power_drill`、`scissors`、`adjustable_wrench`、`hammer`、`baseball`，表面积 `109 cm^2` 至 `643 cm^2`，而单个 DIGIT 传感器视场仅约 `6 cm^2`）。
   - **仿真子集（Sim）：** 在 **TACTO** 中沿长度固定为 `L = 0.5 m` 的测地线（geodesic）随机路径滑行生成，每物体 5 条轨迹共 **50 条仿真滑动序列**，对位姿叠加高斯噪声（`σ_trans = 0.5 mm`, `σ_rot = 1°`），提供 DIGIT RGB 图像（`240×320`）、带噪/真值位姿、`gt_heightmaps`、`gt_contactmasks` 与 YCB 真值网格。
   - **真实子集（Real）：** 将 YCB 物体用重型台钳固定，人工手持 **DIGIT** 沿表面滑动 **60 秒（30 Hz）**，每物体 5 条轨迹共 **50 条真实滑动序列**。采用 8 相机 **OptiTrack** 动捕系统（DIGIT 贴 6 个反光标记点，物体贴 4 个）记录位姿，并对 3 颗 DIGIT 传感器（各自 10 张空载背景校准图）及机械装配偏差做后处理对齐（`alignment.npy`）。**真值高度图与接触掩膜仅仿真子集提供**，真实子集提供 DIGIT 帧、同步位姿 `synced_data.npy` 与旁观 webcam 图像/视频。
2. **MidasTouch 滑动触觉定位方法（约 10 Hz）：**
   - **Tactile Depth Network (TDN)：** ResNet-50 FCRN 在 40 个保留训练集 YCB 物体 × 5,000 次 TACTO 随机压入（共 20 万张合成图像）上训练，高度图重建 RMSE 达 **0.135 mm**，并结合阈值与点云采样生成局部三维接触几何 `P_t`。
   - **Tactile Code Network (TCN)：** 基于 **MinkLoc3D**（稀疏 3D 卷积 FPN + GeM 池化），先在 LiDAR 数据集预训练再以三元组损失微调为 256 维旋转/局部几何嵌入 `E_t ∈ R^256`，用于构建单物体预计算**触觉码本（tactile codebook）**（约 `M = 50,000` 个采样位姿与特征码，配合 `α=0.01` 旋转加权的 `SE(3)` KD-tree 检索，存储体积较图像特征法小约 300 倍）。
   - **蒙特卡洛粒子滤波（PF）：** 初始 `N_0 = 50,000` 个全表面粒子随滑动序列迭代更新：基于最近 `w` 步轨迹局部码本相似度的观测模型、基于里程计不确定性的低方差重采样（动态缩减粒子数至最低 `1,000`）、2 mm 网格最近邻表面投影剪枝以及 DBSCAN 聚类，最终输出多峰 `SE(3)` 位姿假设集合。
3. **关键定量结果（各设定 500 次试验）：**
   - **Sim 集：** 平均最终粒子位置误差 `0.74 cm`、姿态误差 `9.43°`（最接近真值聚类中位误差 `0.28 cm`、`2.03°`）；
   - **Real 集：** 收紧局部先验尺度（`β = 0.5`）并对高度图做指数时间平滑、每 5 步重采样后，平均最终粒子 RMSE `1.97 cm`、`21.48°`（最接近聚类的中位误差 `1.11 cm`、`10.76°`）；
   - **失效边界：** 完全对称且无几何纹理特征的 `baseball` 无法收敛至单峰唯一位姿；`sugar_box`、`mustard_bottle`、`mug` 等具有平面/轴对称性的物体会保留合理的多峰假设分布。

---

## 仓库代码核对

- `download_dataset.sh`：通过 `gdown` 拉取 `sim`、`real`、DIGIT 背景与 `obj_models.zip` 并解压。
- `ycb_slide/record/record_data.py`：调用 `digit_interface.Digit` 与 OpenCV 多线程 `webcam_thread` 以 30 Hz 录制 `frames/`、`webcam_frames/` 及 `digit_data.npy`（含相对时间戳）。
- `ycb_slide/record/timesync_data.py` & `align_data.py`：解析 OptiTrack CSV（`parse_optitrack_csv`）与 DIGIT 帧时间戳同步生成 `synced_data.npy`，再手动微调刚体相对变换并写入 `alignment.npy`。
- `ycb_slide/viz/visualize_trajectory.py` & `visualize_tactile.py`：分别在 YCB `nontextured.stl` 网格上可视化 `SE(3)` 滑动位姿（real 模式读取 `synced_data.npy` + `alignment.npy`；sim 模式读取 `tactile_data.pkl`），以及同屏回放 DIGIT 触觉图与仿真真值高度/接触掩膜。

---

## 映射到 Wiki

- [dataset-ycb-slide.md](../../wiki/entities/dataset-ycb-slide.md)
