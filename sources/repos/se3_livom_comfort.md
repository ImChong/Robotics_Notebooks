# SE(3)-LIVOM for the COMFORT Localization Benchmark

- **类型：** repo
- **仓库：** <https://github.com/url-kaist/se3-livom-comfort>
- **论文：** [arXiv:2611.22047](../papers/se3_livom_arxiv_2611_22047.md)
- **许可：** GPL-2.0（仓库标注）
- **数据：** [GrandTour / COMFORT](https://grand-tour.leggedrobotics.com/)
- **核查日期：** 2026-11-24
- **开放状态：** 已开源，包含 Docker、运行脚本、Python/C++ 实现和评测工具；测试场景真值由 benchmark 服务隐藏。

## 运行入口与核查
- [scripts/run.sh](https://github.com/url-kaist/se3-livom-comfort/blob/master/scripts/run.sh)：选择 multi、ad-n2000、在线 bag、4 前端线程；默认额外启用 4 后端线程。
- [scripts/run_se3lio.sh](https://github.com/url-kaist/se3-livom-comfort/blob/master/scripts/run_se3lio.sh)：读取每任务 IMU 偏移、准备配置及 Docker，检查/构建 pybind 后调用 tools/se3lio_run.py；检查完成标志后转换棱镜坐标。
- [tools/se3lio_run.py](https://github.com/url-kaist/se3-livom-comfort/blob/master/tools/se3lio_run.py)：read_calib、ComfortBagDataset、可选 _OnlineBackend、OdometryPipeline.run，保存轨迹与 timing.csv、latency.csv、backend_rounds.csv。
- README: `bash docker/build_docker.sh`；设置 `DATA` 指向 GrandTour mission 目录；`bash scripts/run.sh arc-2 frontend` 或 `bash scripts/run.sh arc-2`。评测使用 tools/eval.py；测试真值不可本地获取。

## Wiki 映射
- [SE(3)-LIVOM](../../wiki/entities/paper-se3-livom.md)
