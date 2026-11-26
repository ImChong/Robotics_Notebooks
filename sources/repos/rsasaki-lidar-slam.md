# lidar_slam — LiDAR 地图制作与跨会话地图维护（一手源码归档）

- **类型：** repo；不是独立论文，也不是 MAP IV Engine。
- **维护者：** Ryohei Sasaki（[作者主页](https://rsasaki0109.github.io/)自述在 MAP IV，TIER IV group 工作；本仓库由个人账号发布）。
- **原文：** [官方 README](https://github.com/rsasaki0109/lidar_slam/blob/bfe6d280e06cf6433f714a70f8e0f725a274890c/README.md)。
- **核对版本：** `bfe6d280e06cf6433f714a70f8e0f725a274890c`（2026-11-25 上游快照）。
- **开源许可：** [仓库 BSD-2-Clause](https://github.com/rsasaki0109/lidar_slam/blob/bfe6d280e06cf6433f714a70f8e0f725a274890c/LICENSE)；README 标注默认前端 RKO-LIO 为 MIT。数据集另见各自许可。
- **核对范围：** 阅读官方 README 全文、媒体示例的 NCLT 复现说明、长程路线的 NCLT 协议与留出实验、Autoware Quickstart 的常规与高级入口；检查 NCLT 脚本和全部 19 条日志、Python CLI 分发、向导/后端/验证入口以及 live 会话的重定位和提交逻辑。未运行建图或真车。
- **通读：** 已完成
- **入库日期：** 2026-11-26
- **对应节点：** [lidar_slam](../../wiki/entities/rsasaki-lidar-slam.md)

## 原文及证据入口

- [演示生成口径](https://github.com/rsasaki0109/lidar_slam/blob/bfe6d280e06cf6433f714a70f8e0f725a274890c/docs/readme-media.md#one-map-kept-for-a-year-docsassetsimageslifelong_ncltgif)
- [长程地图实验说明](https://rsasaki0109.github.io/lidar_slam/roadmap/long-term-mapping-2026-11.html) · [串联回放脚本与逐次日志](https://github.com/rsasaki0109/lidar_slam/tree/bfe6d280e06cf6433f714a70f8e0f725a274890c/bench_report/data/long_term_2026-11/lifelong)
- [作者 Autoware Quickstart](https://github.com/rsasaki0109/lidar_slam/blob/bfe6d280e06cf6433f714a70f8e0f725a274890c/docs/autoware-quickstart.md) · [Autoware 官方文档的开源 SLAM 列表](https://docs.autoware.org/main/tutorials/integrating-autoware/creating-maps/open-source-slam/)
- [密歇根大学 NCLT 数据与原始论文入口](https://robots.engin.umich.edu/nclt/)，数据按上游说明为 ODbL 1.0；数据集论文不等于本软件的发布论文。

## 事实摘录

- 回放覆盖 2012-01-15 到 12-01，使用一个约 150 m × 150 m 校园区域的通过片段（前后各加 10 s）；1 月 15 日是地图种子。到 12 月 1 日，日志有 18 个无人工首姿态复访条目 `exit: 0` 且有 anchor，锚定耗时约 3.4–159.6 秒**数据时间**。19 是演示会话数，并非成功重定位数。
- 2013-02-23 额外条目为 `exit: 137`，`anchor_after_s: null`。演示说明称已提交会话，但约 14 GB 内存下合并导出未能重写，因此不计入成功锚定和演示导出。
- `lidarslam-map start` 是安装后的工作流入口：先诊断和确认输入与标定，再执行建图。常规路线可输出点云分片、投影信息及轨迹生成的 Lanelet2；高级 RViz/dogfood Quickstart 的范围只是点云，不承诺 Lanelet2。地图验证日志不能代替上线验证。
