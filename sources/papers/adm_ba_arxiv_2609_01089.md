# Adaptive Depth-Map-Guided Bundle Adjustment for Correspondence-Free Multi-View Point Cloud Registration（arXiv:2609.01089）

> 来源归档（ingest）

- **标题：** Adaptive Depth-Map-Guided Bundle Adjustment for Correspondence-Free Multi-View Point Cloud Registration
- **短名：** ADM-BA
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.01089>
- **PDF：** <https://arxiv.org/pdf/2609.01089>
- **代码：** <https://github.com/YiranZhou-Robotics/ADM-BA>
- **机构：** 悉尼科技大学（UTS）；爱丁堡大学（University of Edinburgh）
- **入库日期：** 2026-09-02
- **索引来源：** [具身智能小站 7 篇盘点](../blogs/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md)
- **一句话说明：** 无对应点多视角点云配准：全局 2.5-D 分层深度图 + softmax 层分配 + 非线性 BA；面向废钢工业场景。

## 开源状态（步骤 2.5，2026-09-02）

- **已开源**：`YiranZhou-Robotics/ADM-BA`。

## 核心摘录（面向 wiki 编译）

### 摘录 1：方法要点

- 光滑金属表面、重复结构与遮挡下传统特征对应易错；深度观测直接投影到全局地图形成约束。
- **对 wiki 的映射：** [paper-adm-ba](../../wiki/entities/paper-adm-ba.md)

### 摘录 2：工业动机

- 废钢处理需密集 3D 测量以估计尺寸、切割区域与避碰火炬路径；重建误差直接影响下游规划。
- **对 wiki 的映射：** [paper-adm-ba](../../wiki/entities/paper-adm-ba.md)

## 当前提炼状态

- [x] 仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-adm-ba.md` 新建
