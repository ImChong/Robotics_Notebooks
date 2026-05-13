# Manipulation Perception（操作 / 抓取感知工具）

面向 **pick-and-place、bin picking、动态抓取** 等任务中「从视觉/深度到抓取位姿」这一层的开源与半开源工具索引。适合已理解 **[Manipulation](../../wiki/tasks/manipulation.md)** 任务划分、需要选型 **检测式抓取** 或 **SDK 集成** 的读者。

## 稠密抓取与跟踪

- **[AnyGrasp](../../wiki/entities/anygrasp.md)**（[SDK](https://github.com/graspnet/anygrasp_sdk) · [论文](https://arxiv.org/abs/2212.08333) · [项目页](https://graspnet.net/anygrasp.html)）  
  **7-DoF 稠密抓取 + 跨帧关联**；**预编译库 + License**，实现参考 [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)。

## 数据与评测接口

- **GraspNet 数据集与文档**：https://graspnet.net/datasets.html  
- **graspnetAPI**（Python 评测/数据接口）：https://github.com/graspnet/graspnetAPI  

## 原始资料入口（ingest 溯源）

- [sources/repos/anygrasp-sdk.md](../../sources/repos/anygrasp-sdk.md)
