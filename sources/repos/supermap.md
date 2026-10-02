# SuperMap 官方仓库

- **标题：** SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）AirLab
- **作者：** Shibo Zhao、Guofei Chen、Honghao Zhu、Zhiheng Li、Changwei Yao、Nader Zantout、Seungchan Kim、Wenshan Wang、Ji Zhang、Sebastian Scherer
- **会议：** RSS 2026
- **arXiv：** <https://arxiv.org/abs/2608.22896>
- **论文：** <https://www.roboticsproceedings.org/rss22/p052.pdf>
- **项目页：** <https://superodometry.com/supermap>
- **代码入口：** <https://github.com/superxslam/SuperMap>
- **核查日期：** 2026-10-02
- **开放状态：** 待发布可运行源码；当前公开仓库为 README 与 doc 文档，README 仍注明代码将在 RSS 后发布。项目页宣称 open-source，不能据此认定实现已发布。

## 当前公开内容与复现边界

可见顶层为 `README.md` 与 `doc/`。README 描述离线入口 `examples/example.py`、数据准备脚本和 ROS2 `semantic_mapping` 包，但当前源码树未提供这些实现；这些只能作为未来接口说明，不是已验证可执行路径。未发现 LICENSE 文件，不能推定商用许可。

README 声明环境：Ubuntu 22.04/24.04、Python ≥3.10、NVIDIA GPU ≥16 GB 显存，在线模式 ROS2 Jazzy。输入 RGB、CameraInfo、PointCloud2、Odometry；描述输出物体点云、标注框与带 id/关系/状态/时间戳的 JSON。以上均为文档声明，未运行核验。

## 关联归档

- [项目页](../sites/supermap.md)
- [论文](../papers/supermap_arxiv_2608_22896.md)
- [知识提炼](../../wiki/entities/paper-supermap.md)
