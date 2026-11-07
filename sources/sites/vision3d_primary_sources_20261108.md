# 视觉感知、SLAM 与三维重建原始资料索引（2026-11-08）

> 来源：用户给出的 10 组研究目录。排除 3D视觉工坊课程链接；保留作者论文、作者仓库、官方项目主页与官方文档。
> 状态：**来源入库草稿**；批次中的论文和代码未全部完成通篇精读，故不能声称逐项目知识节点与时序图已经完成。

- **去重后的一手资料 URL：** 90
- **项目/方法身份组：** 63
- **复用已有资料档案：** 4
- **新增逐项目来源档案：** 59
- **通读：** 未完成
- **核对范围：** 已核查仓库 ingest 规范、现有项目路径，及代表性 VINS-Fusion、LVI-SAM、3DGS 的公开原始描述；其余论文需逐篇核对完整正文及附录。

## 去重与分类说明

- VINS-Fusion、ORB-SLAM3、Autoware、LIO-SAM 的既有来源复用；PointNet 的既有论文索引不重复生成 wiki 节点。
- 项目主页、论文、代码可能是同一项目的不同材料，不能仅按 URL 数量建立不同的 wiki 实体。
- OED（dynamic scene graph generation）不是 3DGS；FeatDepth 候选代码作者身份和 GaussianMesh 中文映射尚待确认。
- 截图中的基础方法/歧义标题（例如部分结构光组合编码、Complex-YOLO）没有唯一原始论文，不凭猜测入库。

## 01｜视觉惯性 SLAM

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| VINS-Fusion（视觉惯性 SLAM） | [主来源](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) | [复用](../../sources/repos/vins_fusion.md) |
| VINS-Mono（单目视觉惯性里程计） | [主来源](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) | [初编](../../sources/repos/vision3d_vins_mono.md) |
| ORB-SLAM3（视觉/惯性 SLAM） | [主来源](https://github.com/UZ-SLAMLab/ORB_SLAM3) | [复用](../../sources/repos/orb_slam3.md) |
| Ceres Solver（非线性最小二乘） | [主来源](https://github.com/ceres-solver/ceres-solver) | [初编](../../sources/repos/vision3d_ceres_solver.md) |
| GTSAM（因子图与平滑估计） | [主来源](https://github.com/borglab/gtsam) | [初编](../../sources/repos/vision3d_gtsam.md) |
| Kalibr（相机与 IMU 时空标定） | [主来源](https://github.com/ethz-asl/kalibr)、[配套来源2](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration) | [初编](../../sources/repos/vision3d_kalibr.md) |

## 02｜多传感器融合

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| KITTI（数据与 3D 检测/深度基准） | [主来源](https://www.cvlibs.net/datasets/kitti/index.php)、[配套来源2](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)、[配套来源3](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) | [初编](../../sources/sites/vision3d_kitti.md) |
| Autoware（自动驾驶传感器标定与融合文档） | [主来源](https://docs.autoware.org/main/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/calibrating-sensors/)、[配套来源2](https://docs.autoware.org/main/tutorials/integrating-autoware/integrating-sensors/integrating-cameras/)、[配套来源3](https://docs.autoware.org/main/design/autoware-architecture-v1/components/perception/reference-implementations/radar-based-3d-detector/radar-based-3d-detector/) | [复用](../../sources/repos/autoware.md) |
| PointPainting（相机与点云融合检测） | [主来源](https://openaccess.thecvf.com/content_CVPR_2020/html/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.html) | [初编](../../sources/papers/vision3d_pointpainting.md) |

## 03｜点云目标检测

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| PointNet（点集神经网络） | [主来源](https://web.stanford.edu/~rqi/pointnet/)、[配套来源2](https://github.com/charlesq34/pointnet) | [初编](../../sources/repos/vision3d_pointnet.md) |
| PointNet++（分层点集学习） | [主来源](https://github.com/charlesq34/pointnet2) | [初编](../../sources/repos/vision3d_pointnet2.md) |
| OpenPCDet（含 PointRCNN 与 PV-RCNN） | [主来源](https://github.com/open-mmlab/OpenPCDet) | [初编](../../sources/repos/vision3d_openpcdet.md) |
| VoxelNet（体素化三维检测） | [主来源](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html) | [初编](../../sources/papers/vision3d_voxelnet.md) |
| SECOND（稀疏卷积三维检测） | [主来源](https://www.mdpi.com/1424-8220/18/10/3337) | [初编](../../sources/papers/vision3d_second.md) |
| PointPillars（柱状点云检测） | [主来源](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html) | [初编](../../sources/papers/vision3d_pointpillars.md) |
| 3DSSD（单阶段 3D 检测） | [主来源](https://github.com/JIA-Lab-research/3DSSD) | [初编](../../sources/repos/vision3d_3dssd.md) |
| SA-SSD（辅助监督 3D 检测） | [主来源](https://github.com/skyhehe123/SA-SSD) | [初编](../../sources/repos/vision3d_sa_ssd.md) |
| CIA-SSD（三维目标检测） | [主来源](https://arxiv.org/abs/2012.03015)、[配套来源2](https://github.com/Vegeta2020/CIA-SSD) | [初编](../../sources/repos/vision3d_cia_ssd.md) |

## 04｜单目深度估计

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| SfMLearner（自监督深度与相机运动） | [主来源](https://github.com/tinghuiz/SfMLearner)、[配套来源2](https://research.google/pubs/unsupervised-learning-of-depth-and-ego-motion-from-video/) | [初编](../../sources/repos/vision3d_sfmlearner.md) |
| FeatDepth（单目深度估计） | [主来源](https://arxiv.org/abs/2007.10603)、[配套来源2](https://github.com/sconlyshootery/FeatDepth) | [初编](../../sources/repos/vision3d_featdepth.md) |
| Depth and Motion Learning（深度与运动学习） | [主来源](https://github.com/google-research/google-research/tree/master/depth_and_motion_learning)、[配套来源2](https://arxiv.org/abs/2010.16404) | [初编](../../sources/repos/vision3d_depth_motion_learning.md) |
| Monodepth2（单目自监督深度） | [主来源](https://github.com/nianticlabs/monodepth2) | [初编](../../sources/repos/vision3d_monodepth2.md) |
| COLMAP（SfM 与多视图立体重建） | [主来源](https://github.com/colmap/colmap) | [初编](../../sources/repos/vision3d_colmap.md) |

## 05｜激光-视觉-惯性 SLAM

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| LVI-SAM（LiDAR-视觉-惯性耦合 SLAM） | [主来源](https://arxiv.org/abs/2104.10831)、[配套来源2](https://github.com/TixiaoShan/LVI-SAM) | [初编](../../sources/repos/vision3d_lvi_sam.md) |
| LIO-SAM（激光惯性 SLAM） | [主来源](https://github.com/TixiaoShan/LIO-SAM) | [复用](../../sources/repos/lio_sam.md) |

## 06｜线结构光

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| Steger（线结构光中心线提取） | [主来源](https://ieeexplore.ieee.org/document/659930/) | [初编](../../sources/papers/vision3d_steger_line.md) |
| Zhang（平面相机标定法） | [主来源](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/) | [初编](../../sources/papers/vision3d_zhang_calibration.md) |

## 07｜面结构光

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| 3DUNDERWORLD（结构光扫描仪） | [主来源](https://github.com/theICTlab/3DUNDERWORLD-Structured-Light-Scanner)、[配套来源2](https://3dunderworld.org/software/) | [初编](../../sources/repos/vision3d_3dunderworld.md) |
| Three-Step Phase Shifting（三步相移） | [主来源](https://engineering.purdue.edu/ZhangLab/publications/papers/2006-ao-f3step.pdf) | [初编](../../sources/papers/vision3d_three_step_phase.md) |

## 08｜相机标定

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| OpenCV（相机标定官方文档） | [主来源](https://docs.opencv.org/doc/doxygen/html/d4/d93/group__calib.html)、[配套来源2](https://docs.opencv.org/5.0/tutorials/calib3d/camera_multiview_calibration/multiview_calibration.html)、[配套来源3](https://docs.opencv.org/5.0/tutorials_contrib/ccalib/omnidir_tutorial.html) | [初编](../../sources/sites/vision3d_opencv_calibration.md) |
| Kannala–Brandt（鱼眼相机模型） | [主来源](https://users.aalto.fi/~kannalj1/publications/tpami2006.pdf) | [初编](../../sources/papers/vision3d_kannala_brandt.md) |

## 09｜NeRF 与神经 SLAM

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| NeRF（神经辐射场） | [主来源](https://github.com/bmild/nerf) | [初编](../../sources/repos/vision3d_nerf.md) |
| iMAP（实时隐式建图） | [主来源](https://openaccess.thecvf.com/content/ICCV2021/html/Sucar_iMAP_Implicit_Mapping_and_Positioning_in_Real-Time_ICCV_2021_paper.html) | [初编](../../sources/papers/vision3d_imap.md) |
| NICE-SLAM（神经隐式 SLAM） | [主来源](https://github.com/cvg/nice-slam) | [初编](../../sources/repos/vision3d_nice_slam.md) |
| 3D Gaussian Splatting（原始 Gaussian 渲染） | [主来源](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、[配套来源2](https://github.com/graphdeco-inria/gaussian-splatting) | [初编](../../sources/repos/vision3d_3dgs.md) |
| SplaTAM（Gaussian RGB-D SLAM） | [主来源](https://github.com/spla-tam/SplaTAM) | [初编](../../sources/repos/vision3d_splatam.md) |
| MonoGS（Gaussian SLAM） | [主来源](https://github.com/VSLAM-LAB/MonoGS) | [初编](../../sources/repos/vision3d_monogs.md) |
| Gaussian-SLAM（高斯表示 SLAM） | [主来源](https://github.com/VladimirYugay/Gaussian-SLAM) | [初编](../../sources/repos/vision3d_gaussian_slam.md) |
| GS-SLAM（稠密视觉 Gaussian SLAM） | [主来源](https://gs-slam.github.io/)、[配套来源2](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html) | [初编](../../sources/papers/vision3d_gs_slam.md) |

## 10｜Gaussian Splatting 与动态重建

| 项目（中文 / English） | 一手链接 | 本仓库资料档案 |
|---|---|---|
| COLMAP-Free 3DGS（无需外部 COLMAP 位姿的 Gaussian 建模） | [主来源](https://github.com/NVlabs/CF-3DGS) | [初编](../../sources/repos/vision3d_cf_3dgs.md) |
| GaussianPro（渐进传播 Gaussian） | [主来源](https://proceedings.mlr.press/v235/cheng24f.html) | [初编](../../sources/papers/vision3d_gaussianpro.md) |
| EWA Volume Splatting（体渲染历史方法） | [主来源](https://vcg.seas.harvard.edu/publications/20011001-ewa-volume-splatting) | [初编](../../sources/papers/vision3d_ewa_volume.md) |
| Surface Splatting（表面点片渲染） | [主来源](https://vcg.seas.harvard.edu/publications/20010101-surface-splatting) | [初编](../../sources/papers/vision3d_surface_splatting.md) |
| LightGaussian（高斯表示压缩） | [主来源](https://github.com/VITA-Group/LightGaussian) | [初编](../../sources/repos/vision3d_lightgaussian.md) |
| BAD-Gaussians（去模糊与束调整） | [主来源](https://github.com/WU-CVGL/BAD-Gaussians) | [初编](../../sources/repos/vision3d_bad_gaussians.md) |
| Spec-Gaussian（高光外观建模） | [主来源](https://github.com/ingra14m/Specular-Gaussians) | [初编](../../sources/repos/vision3d_spec_gaussian.md) |
| GeoGaussian（几何感知 Gaussian） | [主来源](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5218_ECCV_2024_paper.php)、[配套来源2](https://yanyan-li.github.io/project/gs/geogaussian) | [初编](../../sources/papers/vision3d_geogaussian.md) |
| Street Gaussians（动态街景） | [主来源](https://arxiv.org/abs/2401.01339) | [初编](../../sources/papers/vision3d_street_gaussians.md) |
| 4D Gaussian Splatting（动态高斯表示） | [主来源](https://github.com/hustvl/4DGaussians)、[配套来源2](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) | [初编](../../sources/repos/vision3d_4dgaussians.md) |
| 3D Geometry-Aware Deformable Gaussian Splatting（动态几何） | [主来源](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_3D_Geometry-Aware_Deformable_Gaussian_Splatting_for_Dynamic_View_Synthesis_CVPR_2024_paper.html) | [初编](../../sources/papers/vision3d_geometry_aware_deformable.md) |
| Deformable 3D Gaussians（单目动态重建） | [主来源](https://github.com/ingra14m/Deformable-3D-Gaussians)、[配套来源2](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.html) | [初编](../../sources/repos/vision3d_deformable_gaussians.md) |
| 2D Gaussian Splatting（表面重建） | [主来源](https://github.com/hbb1/2d-gaussian-splatting)、[配套来源2](https://surfsplatting.github.io/) | [初编](../../sources/repos/vision3d_2d_gaussian.md) |
| SuGaR（Gaussian 表面几何提取） | [主来源](https://anttwo.github.io/sugar/) | [初编](../../sources/sites/vision3d_sugar.md) |
| 4D-Rotor Gaussian Splatting（动态重建） | [主来源](https://research.nvidia.com/publication/2024-02_4d-rotor-gaussian-splatting-towards-efficient-novel-view-synthesis-dynamic)、[配套来源2](https://github.com/weify627/4D-Rotor-Gaussians) | [初编](../../sources/repos/vision3d_4d_rotor.md) |
| Splatter Image（单视图快速三维重建） | [主来源](https://github.com/szymanowiczs/splatter-image)、[配套来源2](https://openaccess.thecvf.com/content/CVPR2024/html/Szymanowicz_Splatter_Image_Ultra-Fast_Single-View_3D_Reconstruction_CVPR_2024_paper.html) | [初编](../../sources/repos/vision3d_splatter_image.md) |
| Triplane Meets Gaussian Splatting（单视图重建） | [主来源](https://github.com/VAST-AI-Research/TriplaneGaussian)、[配套来源2](https://openaccess.thecvf.com/content/CVPR2024/html/Zou_Triplane_Meets_Gaussian_Splatting_Fast_and_Generalizable_Single-View_3D_Reconstruction_CVPR_2024_paper.html) | [初编](../../sources/repos/vision3d_triplane_gaussian.md) |
| GaussianMesh（高斯网格形变，题名待核） | [主来源](https://github.com/IGLICT/GaussianMesh) | [初编](../../sources/repos/vision3d_gaussianmesh.md) |
| GauFRe（Gaussian Deformation Fields） | [主来源](https://arxiv.org/abs/2312.11458) | [初编](../../sources/papers/vision3d_gaufre.md) |
| mip-NeRF（抗锯齿神经辐射场） | [主来源](https://github.com/google/mipnerf)、[配套来源2](https://openaccess.thecvf.com/content/ICCV2021/html/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.html) | [初编](../../sources/repos/vision3d_mip_nerf.md) |
| Zip-NeRF（抗锯齿网格辐射场） | [主来源](https://openaccess.thecvf.com/content/ICCV2023/html/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.html) | [初编](../../sources/papers/vision3d_zip_nerf.md) |
| GS-Pose（Gaussian 物体位姿估计） | [主来源](https://dingdingcai.github.io/gs-pose)、[配套来源2](https://github.com/dingdingcai/GSPose) | [初编](../../sources/repos/vision3d_gs_pose.md) |
| BAGS（模糊感知 Gaussian Splatting） | [主来源](https://nwang43jhu.github.io/BAGS/)、[配套来源2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10394_ECCV_2024_paper.php) | [初编](../../sources/papers/vision3d_bags.md) |
| OED（动态场景图生成，非 3DGS） | [主来源](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_OED_Towards_One-stage_End-to-End_Dynamic_Scene_Graph_Generation_CVPR_2024_paper.html)、[配套来源2](https://github.com/guanw-pku/OED) | [初编](../../sources/repos/vision3d_oed.md) |

## 质量审计（未完成）

- [ ] 所有主论文、项目原始资料逐项通读（方法、实验、限制、必要附录）。
- [ ] 仓库/项目主页逐项核验代码、权重与数据开放范围。
- [ ] 新建或实质更新的 wiki 实体按原文补 Mermaid 主流程，按源码真实入口补 sequenceDiagram。
- [ ] 运行 preflight、tests 和 report_health；未通过之前本 PR 仅作为草稿来源归档。
