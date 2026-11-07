# 视觉 SLAM／点云检测／深度估计／结构光／NeRF／3DGS：一手来源核对索引

> 2026-11-08 ingest；来自用户给定的 10 张课程目录的**研究成果溯源**，不归档、不引用课程提供方 3D视觉工坊自身的课程/视频/商品资料。

- **类型：** 一手资料来源清单（论文 / 官方代码 / 项目页 / 官方技术文档）
- **去重后原始链接：** 90 条
- **核对范围：** 对照课程截图主题；已核对 GitHub 原作者仓库的部分元信息及 LVI-SAM、FeatDepth README。其余多数论文正文、附录与源码入口尚待全文核验。
- **通读：** 未完成
- **重要边界：** 本页是 source 层覆盖清单，**不是** 90 篇原文均已精读的声明；不得据此自动将实体页标为 complete。任何节点升格必须先通读原文，补原文依据的 Mermaid 数据流；有可运行官方代码时必须核对启动入口并给时序图。

## 已核对的关键身份

- **FeatDepth**：`sconlyshootery/FeatDepth` README 明确称为 ECCV 2020《Feature-metric Loss for Self-supervised Learning of Depth and Egomotion》的官方实现，包含 `train.py`、`run.py` 与 `scripts/infer.py` 等入口。
- **LVI-SAM**：`TixiaoShan/LVI-SAM` README 说明激光-惯性 LIS 与视觉-惯性 VIS 的组合、ROS `roslaunch lvi_sam run.launch` 入口。
- **OpenPCDet**：为 PointRCNN/PV-RCNN 等项目的实现集合；单一仓库不等于这两个方法是同一个研究工作。
- **OED**：链接实际研究动态场景**图**生成，与 3D Gaussian Splatting 不是同一领域，应单列，不按 3DGS 项目建立节点。

## 来源逐条映射

| 编号 | 对应方向 | 原始资料 | 类型 |
|---|---|---|---|
| 01 | 视觉惯性SLAM与VINS-Fusion | [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) | 官方代码/仓库 |
| 02 | 视觉惯性SLAM与VINS-Fusion | [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) | 官方代码/仓库 |
| 03 | 视觉惯性SLAM与VINS-Fusion | [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | 官方代码/仓库 |
| 04 | 视觉惯性SLAM与VINS-Fusion | [Ceres Solver](https://github.com/ceres-solver/ceres-solver) | 官方代码/仓库 |
| 05 | 视觉惯性SLAM与VINS-Fusion | [GTSAM](https://github.com/borglab/gtsam) | 官方代码/仓库 |
| 06 | 视觉惯性SLAM与VINS-Fusion | [Kalibr](https://github.com/ethz-asl/kalibr) | 官方代码/仓库 |
| 07 | 视觉惯性SLAM与VINS-Fusion | [Kalibr Camera-IMU calibration](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration) | 官方使用文档 |
| 08 | 视觉惯性SLAM与VINS-Fusion | [KITTI dataset](https://www.cvlibs.net/datasets/kitti/index.php) | 官方项目/技术文档 |
| 09 | 自动驾驶多传感器融合 | [Autoware 传感器标定](https://docs.autoware.org/main/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/calibrating-sensors/) | 官方项目/技术文档 |
| 10 | 自动驾驶多传感器融合 | [Autoware 相机集成](https://docs.autoware.org/main/tutorials/integrating-autoware/integrating-sensors/integrating-cameras/) | 官方项目/技术文档 |
| 11 | 自动驾驶多传感器融合 | [Autoware 雷达检测](https://docs.autoware.org/main/design/autoware-architecture-v1/components/perception/reference-implementations/radar-based-3d-detector/radar-based-3d-detector/) | 官方项目/技术文档 |
| 12 | 自动驾驶多传感器融合 | [PointPainting](https://openaccess.thecvf.com/content_CVPR_2020/html/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.html) | 论文原始发布 |
| 13 | 自动驾驶多传感器融合 | [LVI-SAM 论文](https://arxiv.org/abs/2104.10831) | 论文原始发布 |
| 14 | 自动驾驶多传感器融合 | [LVI-SAM 代码](https://github.com/TixiaoShan/LVI-SAM) | 官方代码/仓库 |
| 15 | 三维点云目标检测 | [PointNet 作者项目](https://web.stanford.edu/~rqi/pointnet/) | 官方项目/技术文档 |
| 16 | 三维点云目标检测 | [PointNet 源码](https://github.com/charlesq34/pointnet) | 官方代码/仓库 |
| 17 | 三维点云目标检测 | [PointNet++ 源码](https://github.com/charlesq34/pointnet2) | 官方代码/仓库 |
| 18 | 三维点云目标检测 | [OpenPCDet（PointRCNN/PV-RCNN）](https://github.com/open-mmlab/OpenPCDet) | 官方代码/仓库 |
| 19 | 三维点云目标检测 | [VoxelNet 原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html) | 论文原始发布 |
| 20 | 三维点云目标检测 | [SECOND 原论文](https://www.mdpi.com/1424-8220/18/10/3337) | 论文原始发布 |
| 21 | 三维点云目标检测 | [PointPillars 原论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html) | 论文原始发布 |
| 22 | 三维点云目标检测 | [3DSSD 源码](https://github.com/JIA-Lab-research/3DSSD) | 官方代码/仓库 |
| 23 | 三维点云目标检测 | [SA-SSD 源码](https://github.com/skyhehe123/SA-SSD) | 官方代码/仓库 |
| 24 | 三维点云目标检测 | [CIA-SSD 原论文](https://arxiv.org/abs/2012.03015) | 论文原始发布 |
| 25 | 三维点云目标检测 | [CIA-SSD 源码](https://github.com/Vegeta2020/CIA-SSD) | 官方代码/仓库 |
| 26 | 三维点云目标检测 | [KITTI 3D 检测评测](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) | 官方项目/技术文档 |
| 27 | 单目深度估计 | [SfMLearner 源码](https://github.com/tinghuiz/SfMLearner) | 官方代码/仓库 |
| 28 | 单目深度估计 | [SfMLearner 原论文](https://research.google/pubs/unsupervised-learning-of-depth-and-ego-motion-from-video/) | 官方项目/技术文档 |
| 29 | 单目深度估计 | [FeatDepth 原论文](https://arxiv.org/abs/2007.10603) | 论文原始发布 |
| 30 | 单目深度估计 | [FeatDepth 源码](https://github.com/sconlyshootery/FeatDepth) | 官方代码/仓库 |
| 31 | 单目深度估计 | [Depth and Motion Learning 源码](https://github.com/google-research/google-research/tree/master/depth_and_motion_learning) | 官方代码/仓库 |
| 32 | 单目深度估计 | [Depth and Motion Learning 论文](https://arxiv.org/abs/2010.16404) | 论文原始发布 |
| 33 | 单目深度估计 | [Monodepth2 源码](https://github.com/nianticlabs/monodepth2) | 官方代码/仓库 |
| 34 | 单目深度估计 | [COLMAP 源码](https://github.com/colmap/colmap) | 官方代码/仓库 |
| 35 | 单目深度估计 | [KITTI 深度评测](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) | 官方项目/技术文档 |
| 36 | 激光-惯性SLAM | [LIO-SAM 源码](https://github.com/TixiaoShan/LIO-SAM) | 官方代码/仓库 |
| 37 | 线结构光 | [Steger 曲线/线中心提取论文](https://ieeexplore.ieee.org/document/659930/) | 论文原始发布 |
| 38 | 线结构光 | [张正友相机标定论文](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/) | 官方项目/技术文档 |
| 39 | 线结构光 | [OpenCV 标定 API 文档](https://docs.opencv.org/doc/doxygen/html/d4/d93/group__calib.html) | 官方项目/技术文档 |
| 40 | 面结构光 | [3DUNDERWORLD 代码](https://github.com/theICTlab/3DUNDERWORLD-Structured-Light-Scanner) | 官方代码/仓库 |
| 41 | 面结构光 | [3DUNDERWORLD 项目页](https://3dunderworld.org/software/) | 官方项目/技术文档 |
| 42 | 面结构光 | [三步相移测量原论文](https://engineering.purdue.edu/ZhangLab/publications/papers/2006-ao-f3step.pdf) | 官方项目/技术文档 |
| 43 | 面结构光 | [OpenCV 多相机标定](https://docs.opencv.org/5.0/tutorials/calib3d/camera_multiview_calibration/multiview_calibration.html) | 官方项目/技术文档 |
| 44 | 相机标定 | [Kannala-Brandt 鱼眼相机模型](https://users.aalto.fi/~kannalj1/publications/tpami2006.pdf) | 官方项目/技术文档 |
| 45 | 相机标定 | [OpenCV 全景相机模型](https://docs.opencv.org/5.0/tutorials_contrib/ccalib/omnidir_tutorial.html) | 官方项目/技术文档 |
| 46 | NeRF 与高斯SLAM | [NeRF 官方实现](https://github.com/bmild/nerf) | 官方代码/仓库 |
| 47 | NeRF 与高斯SLAM | [iMAP 原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Sucar_iMAP_Implicit_Mapping_and_Positioning_in_Real-Time_ICCV_2021_paper.html) | 论文原始发布 |
| 48 | NeRF 与高斯SLAM | [NICE-SLAM 源码](https://github.com/cvg/nice-slam) | 官方代码/仓库 |
| 49 | NeRF 与高斯SLAM | [3DGS 原作者项目](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | 官方项目/技术文档 |
| 50 | NeRF 与高斯SLAM | [3DGS 原作者源码](https://github.com/graphdeco-inria/gaussian-splatting) | 官方代码/仓库 |
| 51 | NeRF 与高斯SLAM | [SplaTAM 源码](https://github.com/spla-tam/SplaTAM) | 官方代码/仓库 |
| 52 | NeRF 与高斯SLAM | [MonoGS 源码](https://github.com/VSLAM-LAB/MonoGS) | 官方代码/仓库 |
| 53 | NeRF 与高斯SLAM | [Gaussian-SLAM 源码](https://github.com/VladimirYugay/Gaussian-SLAM) | 官方代码/仓库 |
| 54 | NeRF 与高斯SLAM | [GS-SLAM 项目页](https://gs-slam.github.io/) | 官方项目/技术文档 |
| 55 | NeRF 与高斯SLAM | [GS-SLAM 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html) | 论文原始发布 |
| 56 | 3DGS与动态三维重建 | [CF-3DGS 源码](https://github.com/NVlabs/CF-3DGS) | 官方代码/仓库 |
| 57 | 3DGS与动态三维重建 | [GaussianPro 原论文](https://proceedings.mlr.press/v235/cheng24f.html) | 论文原始发布 |
| 58 | 3DGS与动态三维重建 | [EWA Volume Splatting 原论文](https://vcg.seas.harvard.edu/publications/20011001-ewa-volume-splatting) | 官方项目/技术文档 |
| 59 | 3DGS与动态三维重建 | [Surface Splatting 原论文](https://vcg.seas.harvard.edu/publications/20010101-surface-splatting) | 官方项目/技术文档 |
| 60 | 3DGS与动态三维重建 | [LightGaussian 源码](https://github.com/VITA-Group/LightGaussian) | 官方代码/仓库 |
| 61 | 3DGS与动态三维重建 | [BAD-Gaussians 源码](https://github.com/WU-CVGL/BAD-Gaussians) | 官方代码/仓库 |
| 62 | 3DGS与动态三维重建 | [Spec-Gaussian 源码](https://github.com/ingra14m/Specular-Gaussians) | 官方代码/仓库 |
| 63 | 3DGS与动态三维重建 | [GeoGaussian 原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5218_ECCV_2024_paper.php) | 论文原始发布 |
| 64 | 3DGS与动态三维重建 | [GeoGaussian 作者项目](https://yanyan-li.github.io/project/gs/geogaussian) | 官方项目/技术文档 |
| 65 | 3DGS与动态三维重建 | [Street Gaussians 原论文](https://arxiv.org/abs/2401.01339) | 论文原始发布 |
| 66 | 3DGS与动态三维重建 | [4D Gaussian Splatting 源码](https://github.com/hustvl/4DGaussians) | 官方代码/仓库 |
| 67 | 3DGS与动态三维重建 | [4D Gaussian Splatting 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) | 论文原始发布 |
| 68 | 3DGS与动态三维重建 | [3D Geometry-Aware Deformable Gaussians 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_3D_Geometry-Aware_Deformable_Gaussian_Splatting_for_Dynamic_View_Synthesis_CVPR_2024_paper.html) | 论文原始发布 |
| 69 | 3DGS与动态三维重建 | [Deformable 3D Gaussians 源码](https://github.com/ingra14m/Deformable-3D-Gaussians) | 官方代码/仓库 |
| 70 | 3DGS与动态三维重建 | [Deformable 3D Gaussians 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.html) | 论文原始发布 |
| 71 | 3DGS与动态三维重建 | [2D Gaussian Splatting 源码](https://github.com/hbb1/2d-gaussian-splatting) | 官方代码/仓库 |
| 72 | 3DGS与动态三维重建 | [2D Gaussian Splatting 作者项目](https://surfsplatting.github.io/) | 官方项目/技术文档 |
| 73 | 3DGS与动态三维重建 | [SuGaR 官方项目](https://anttwo.github.io/sugar/) | 官方项目/技术文档 |
| 74 | 3DGS与动态三维重建 | [4D-Rotor 原论文机构页](https://research.nvidia.com/publication/2024-02_4d-rotor-gaussian-splatting-towards-efficient-novel-view-synthesis-dynamic) | 官方项目/技术文档 |
| 75 | 3DGS与动态三维重建 | [4D-Rotor 源码](https://github.com/weify627/4D-Rotor-Gaussians) | 官方代码/仓库 |
| 76 | 3DGS与动态三维重建 | [Splatter Image 源码](https://github.com/szymanowiczs/splatter-image) | 官方代码/仓库 |
| 77 | 3DGS与动态三维重建 | [Splatter Image 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Szymanowicz_Splatter_Image_Ultra-Fast_Single-View_3D_Reconstruction_CVPR_2024_paper.html) | 论文原始发布 |
| 78 | 3DGS与动态三维重建 | [TriplaneGaussian 源码](https://github.com/VAST-AI-Research/TriplaneGaussian) | 官方代码/仓库 |
| 79 | 3DGS与动态三维重建 | [Triplane Meets Gaussian Splatting 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Zou_Triplane_Meets_Gaussian_Splatting_Fast_and_Generalizable_Single-View_3D_Reconstruction_CVPR_2024_paper.html) | 论文原始发布 |
| 80 | 3DGS与动态三维重建 | [GaussianMesh 源码](https://github.com/IGLICT/GaussianMesh) | 官方代码/仓库 |
| 81 | 3DGS与动态三维重建 | [GauFRe 原论文](https://arxiv.org/abs/2312.11458) | 论文原始发布 |
| 82 | 3DGS与动态三维重建 | [mip-NeRF 源码](https://github.com/google/mipnerf) | 官方代码/仓库 |
| 83 | 3DGS与动态三维重建 | [mip-NeRF 原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.html) | 论文原始发布 |
| 84 | 3DGS与动态三维重建 | [Zip-NeRF 原论文](https://openaccess.thecvf.com/content/ICCV2023/html/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.html) | 论文原始发布 |
| 85 | 3DGS与动态三维重建 | [GS-Pose 作者项目](https://dingdingcai.github.io/gs-pose) | 官方项目/技术文档 |
| 86 | 3DGS与动态三维重建 | [GS-Pose 源码](https://github.com/dingdingcai/GSPose) | 官方代码/仓库 |
| 87 | 3DGS与动态三维重建 | [BAGS 作者项目](https://nwang43jhu.github.io/BAGS/) | 官方项目/技术文档 |
| 88 | 3DGS与动态三维重建 | [BAGS 原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10394_ECCV_2024_paper.php) | 论文原始发布 |
| 89 | 3DGS与动态三维重建 | [OED 原论文（不属3DGS）](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_OED_Towards_One-stage_End-to-End_Dynamic_Scene_Graph_Generation_CVPR_2024_paper.html) | 论文原始发布 |
| 90 | 3DGS与动态三维重建 | [OED 源码（不属3DGS）](https://github.com/guanw-pku/OED) | 官方代码/仓库 |

## 仓库去重与后续实体核验

目前已确认目标仓库已有 VINS-Fusion、ORB-SLAM3、Autoware、LIO-SAM、PointNet、KITTI 及 3DGS（原始论文）等相关节点／归档；不可通过标题重新建同名实体。其他方法须继续核对 arXiv、项目主页、源码身份后再新建或更新对应 wiki 节点。

**未唯一定位：** 课程所写 YOLO3D／Complex-YOLO、部分结构光编码组合、中文意译的动态 3DGS 论文标题；不能凭名称猜原始论文。

**是否完整阅读：否。** 完整论文和源码入口的逐条通读是后续升格实体的前置条件，本索引明确保留此阻塞项。

## 验收核对

- [x] 排除课程提供方原始商品页、视频页
- [x] 去重收录 90 个外部原始研究/文档 URL
- [x] 区分论文、原作者源码、项目/技术文档
- [x] 标明 OED 与 Gaussian Splatting 的主题差异
- [ ] 所有原文全文与官方 README 通读完成
- [ ] 全部独立项目 wiki 节点去重升级与 Mermaid 图核验
- [ ] 本地 `make ci-preflight` + `make ci-test` 与每页健康度报告

## 进一步核对后补充的原始论文（不计入上述 90 条截图索引 URL）

这些论文是原先索引中**只有源码入口或合集入口**但缺失主论文时，依据原作者或论文出版方补找的原文。仍未通读全文，不计为完成实体 ingest。

- **PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud**（CVPR 2019）：[论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html) · [源码](https://github.com/sshaoshuai/PointRCNN) · [单独来源记录](../papers/vision3d_point_rcnn_primary_paper_20261108.md)
- **PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection**（CVPR 2020）：[论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.html) · [源码](https://github.com/open-mmlab/OpenPCDet) · [单独来源记录](../papers/vision3d_pv_rcnn_primary_paper_20261108.md)
- **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**（NeurIPS 2017 / arXiv 1706.02413）：[论文](https://arxiv.org/abs/1706.02413) · [源码](https://github.com/charlesq34/pointnet2) · [单独来源记录](../papers/vision3d_pointnet2_primary_paper_20261108.md)
- **NICE-SLAM: Neural Implicit Scalable Encoding for SLAM**（CVPR 2022）：[论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_NICE-SLAM_Neural_Implicit_Scalable_Encoding_for_SLAM_CVPR_2022_paper.html) · [源码](https://github.com/cvg/nice-slam) · [单独来源记录](../papers/vision3d_nice_slam_primary_paper_20261108.md)
- **SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM**（CVPR 2024）：[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.html) · [源码](https://github.com/spla-tam/SplaTAM) · [单独来源记录](../papers/vision3d_splatam_primary_paper_20261108.md)
- **Gaussian Splatting SLAM**（CVPR 2024）：[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.html) · [源码](https://github.com/VSLAM-LAB/MonoGS) · [单独来源记录](../papers/vision3d_monogs_primary_paper_20261108.md)
- **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**（IEEE Transactions on Robotics 2018）：[论文](https://ieeexplore.ieee.org/document/8421746) · [源码](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) · [单独来源记录](../papers/vision3d_vins_mono_primary_paper_20261108.md)
