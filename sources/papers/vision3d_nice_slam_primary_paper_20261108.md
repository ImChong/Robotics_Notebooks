# NICE-SLAM: Neural Implicit Scalable Encoding for SLAM

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** NICE-SLAM: Neural Implicit Scalable Encoding for SLAM
- **类型：** paper
- **原论文：** https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_NICE-SLAM_Neural_Implicit_Scalable_Encoding_for_SLAM_CVPR_2022_paper.html
- **官方实现：** https://github.com/cvg/nice-slam
- **会议/期刊：** CVPR 2022
- **入库日期：** 2026-11-08
- **一句话说明：** 在稠密 SLAM 中引入多层级局部信息的场景隐式表示，分别处理几何和外观优化以改善大场景重建和跟踪。
- **核对范围：** 已阅读 https://arxiv.org/html/2112.12130 正文、方法、表格、性能分析与附录/局限，核对 https://github.com/cvg/nice-slam 官方源码入口；未独立复现。
- **通读：** 已完成

## 原论文方法与实验事实

粗层 grid 表示墙面与未观测区域的几何概貌，中/细层 grid 优化可见局部细节，另用颜色 grid 做 RGB 重建。Coarse/Mid/Fine 几何采用预训练固定解码 MLP，优化的是局部 feature grid。光线采样和深度/颜色体渲染构造跟踪与建图误差；Tracker 独立优化当前 camera pose，Mapper 优化局部网格和关键帧；大型场景使用可见视锥局部 feature 更新与 keyframe overlap 选取。不是原始 iMAP 的单一全局 MLP。

**实验：** 论文 §4.2/Table2 TUM RGB-D ATE RMSE（cm，作者从 5 次运行取最佳）：NICE-SLAM fr1/desk **2.7**、fr2/xyz **1.8**、fr3/office **3.0**；ORB-SLAM2 对应 1.6/0.4/1.0，不能宣称全局胜传统定位。论文还报告 Replica 重建和 ScanNet 大场景，渲染/跟踪使用不同指标。

**局限：** 几何预测有训练集先验及归纳偏置，跨类别泛化不等于保证；光线/体渲染和双设备 Tracking/Mapping 有算力需求；RGB-D 深度噪声和动态物体造成重建误差，论文中传统特征 SLAM 的部分轨迹指标仍优。

## 方法摘要（以出版方摘要为依据）

在稠密 SLAM 中引入多层级局部信息的场景隐式表示，分别处理几何和外观优化以改善大场景重建和跟踪。

## 源码与论文身份

- 原论文：[NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_NICE-SLAM_Neural_Implicit_Scalable_Encoding_for_SLAM_CVPR_2022_paper.html)
- 代码：[https://github.com/cvg/nice-slam](https://github.com/cvg/nice-slam)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [x] 已核对论文方法/实验/消融及适用边界。
- [x] 核对原作者仓库主程序与 Mapper/Tracker 实现入口。
- [x] 建立 [NICE-SLAM](../../wiki/entities/paper-nice-slam.md) 深读节点。
