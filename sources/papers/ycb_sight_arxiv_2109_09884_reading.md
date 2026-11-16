# Efficient shape mapping through dense touch and vision（YCB-Sight 原文与仓库核对笔记）

- **来源：** <https://arxiv.org/abs/2109.09884>（`arXiv:2109.09884v3`，ICRA 2022，2022-03-10）
- **代码与数据仓库：** <https://github.com/Robo-Touch/YCB-Sight>（文中同时提及 `CMURoboTouch/YCB-Sight` 与重构器 `rpl-cmu/shape-map-3D`）
- **许可证：** 数据集为 CC BY-SA 4.0，配套处理代码为 MIT License
- **作者与机构：** Sudharshan Suresh*, Zilin Si*, Joshua G. Mangelson（BYU）, Wenzhen Yuan, Michael Kaess（CMU）
- **核对日期：** 2026-11-14
- **核对范围：** 论文全文 `arXiv:2109.09884v3`（第 1–1401 行）及 `Robo-Touch/YCB-Sight` 仓库 `README.md`、`LICENSE`、`scripts/lookup_mapping/lookup.py`、`scripts/data_visualization/data_visualizer.py`
- **通读：** 已完成

---

## 开源与数据核查

- **开源状态：** **已开源（按仓库模块化发布）**。`Robo-Touch/YCB-Sight` 提供仿真与真实数据集（Google Drive 分物体打包）、标定文件（`calibrated_models/` 中的 `reconstruct_calib.npz`、`bg.jpg`、`gelmap2.npy`）、查找表高度图重建（`scripts/lookup_mapping/lookup.py`）与数据可视化脚本（`scripts/data_visualization/data_visualizer.py`）；FCRN 神经网络深度预测与 ShapeMap 3-D 主建图管线由独立关联仓库（FCRN-DepthPrediction 及 `rpl-cmu/shape-map-3D`）维护。

---

## 原文核心事实摘录

1. **数据集构成：**
   - **YCBSight-Sim：** 基于 **Taxim**（物理校准的 GelSight 仿真实时渲染器）与 **Pyrender** 俯视深度相机，覆盖 **30 个 YCB 物体**。其中 660 帧/物体的随机位姿样本用于 FCRN 训练与验证（划出 6 个未见评估物体，550/50/60 划分），另为 6 个评估物体（`002_master_chef_can`、`004_sugar_box`、`005_tomato_soup_can`、`010_potted_meat_can`、`021_bleach_cleanser`、`036_wood_block`）各提供 60 次沿表面法向均匀分布的局部按压（最大压深 1 mm，附加零均值高斯位姿/深度噪声），每帧附带真实 `gt_height_map/*.npy`、`gt_contact_mask/*.npy`、`gelsight/*.jpg`、`pose.txt` 及相机深度 `depthCam.npy`。
   - **YCBSight-Real：** 使用 **UR5e** 六轴机械臂 + **WSG50** 并行夹爪搭载 **GelSight** 光学触觉传感器（`640×480` RGB，传感器有效感测面积 `2.66 cm^2`）+ 固定于约 1 m 外、完成手眼标定的 **Azure Kinect** 深度相机，在台钳固定 YCB 物体（上述同款 6 个评估物体）上采集 **每物体 40 次触碰**（约 20 分钟/物体）的 GelSight 图像 `{I_1..I_40}` 与运动学正运动学（FK）位姿 `{p_1..p_40}`，外加初始 Kinect 深度图 `D_0` 与手眼变换 `tf.json`。
2. **ShapeMap 3-D 增量建图管线：**
   - **逆向传感器模型 `Ω: I_t → (H_t, C_t)`：** 基于 ResNet-50 结构的全卷积残差网络（FCRN）直接从仿真配对数据学习局部高度图与二值接触掩膜。相比传统查找表+快速泊松积分，FCRN 在评估集上将高度图 RMSE 从 **0.182 mm** 降至 **0.094 mm**，接触掩膜 IoU 从 **0.379** 提升至 **0.752**。
   - **高斯过程空间图（GP-SG）：** 将 `C_t` 过滤后的局部点云与表面法向投影到世界系，并在 `S^3`（`S=16` 均匀三维查询网格）上使用薄板样条（thin-plate）GP 核函数建模符号距离场（SDF）；通过将全局 `O(N^3)` 高斯过程回归拆分为半径 `r`（物体边长的 `15%`）内的局部一元因子，利用 **GTSAM / iSAM2** 在 CPU（`i7-7820HQ`）上实现实时增量 MAP 推断与在线更新。
   - **表面提取与不确定性：** 提取 SDF 零等值面并通过 Marching Cubes 重建表面网格 `S`（剪除与最近观测量测距离超过 `r` 的区域），同时输出点云/体素级空间方差 `Σ_S`。
3. **关键定量结论：**
   - 仿真环境下 ShapeMap 3-D 在 35–40 次局部按压内即收敛至完整光滑网格；
   - 真实环境 `YCBSight-Real` 约 30 次触碰收敛，跨 6 物体（对角线尺度约 20 cm）平均 Chamfer Distance 达 **18.3 mm^2**；
   - 当物体底部被台钳夹持遮挡、机械臂无法触及底部时，可通过对基座隐式施加虚拟辅助接触点避免建图悬空。

---

## 仓库代码核对

- `scripts/lookup_mapping/lookup.py`（`Tactile2HeightReconstruct` 类）：读取 `calibrated_models/reconstruct_calib.npz`、`bg.jpg` 与 `gelmap2.npy`，执行差分 `dI = img - bg_proc`、增益归一化并查表映射梯度幅值/方向（`match_grad`），最终调用 `scipy.fftpack.dst/idst` 实现 `fast_poisson` 泊松积分并与 `gt_height_map` 对比。
- `scripts/data_visualization/data_visualizer.py`：提供 `visualize_batch_sim`（同步展示触觉图、高度真值与接触掩膜）与 `visualize_batch_real`（按时间戳排序跳过偶数背景帧并展示按压前景帧）。

---

## 映射到 Wiki

- [dataset-ycb-sight.md](../../wiki/entities/dataset-ycb-sight.md)
