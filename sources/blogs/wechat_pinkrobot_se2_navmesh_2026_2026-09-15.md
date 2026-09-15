# 顶刊 2026｜SE(2) 导航网格【文献解读】

> 来源归档（blog / 微信公众号）

- **标题：** 顶刊 2026｜SE(2) 导航网格【文献解读】
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/t0RR3tRB_eWH4tNfNAceWg
- **发表日期：** 2026-09-15（入库日）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch
- **论文：** SE(2) Navigation Mesh
- **arXiv：** <https://arxiv.org/abs/2607.01454>
- **项目页：** <https://se2-navmesh.github.io/>
- **机构：** Robotic Systems Lab, ETH Zürich（文内）；项目页为 Anonymous Submission（截至入库日）
- **一句话说明：** 将 NavMesh 从「圆柱 yaw 无关」升级为 **yaw channel + continuous-yaw footprint mask** 的分层图；**ASA**（A*→String Pulling→A*）联合优化位置与朝向；Voxblox 点云在线 slab 局部更新，ANYmal  onboard 4 Hz。

## 步骤 2.5（开源核查）

- **截至入库日：** 项目页 <https://se2-navmesh.github.io/> **未列 GitHub**；含交互 demo 与 HM3D 场景导出，代码状态 **待发布 / 匿名投稿期**
- **部署证据：** ANYmal + ZED X Mini + Jetson Orin 全 onboard

## 核心摘录（归纳）

### 关键科学问题

同一位置 $p$ 的可通行性应写为 $f(p,\psi)$ 而非 $f(p)$；经典 NavMesh 外接圆把「正着能过、横着不能」的 restricted 区域整块删除。

### 表示三要素

1. **Yaw channel：** 离散 $\psi_i$，每层 $\mathcal{L}_i$ 存可行 footprint
2. **Continuous-yaw mask：** 相邻 channel 间原地旋转的几何安全保证（swept footprint 并集）
3. **两类边：** 平移边（同层邻接）+ 旋转边（跨相邻 yaw layer）

### Safe / Restricted / Inaccessible

- Safe：全部 yaw 可行
- Restricted：部分 yaw 可行（**增益来源**）
- Inaccessible：无可行 yaw

### ASA 三阶段

Initial A*（layer-specific region + 纵/横/转代价）→ String Pulling（只拉直位置）→ Yaw Refinement A*（固定走廊重优化朝向）

### 实验亮点（文内）

- 6 个 HM3D 场景 traversable area **均 +50%**；多边形数 ×8–10 但构建时间仅 ×2–3
- Validity check：SE(2) NM **~0.01 ms** vs voxel checker **~1 ms**
- 真机：0.8 m 窄通道、门洞、悬垂、多楼层楼梯

## 对 wiki 的映射

- **新建：** [paper-se2-navigation-mesh](../../wiki/entities/paper-se2-navigation-mesh.md)
- **交叉：** [Locomotion](../../wiki/tasks/locomotion.md)、[footstep planning 相关概念](../../wiki/concepts/footstep-planning.md)
