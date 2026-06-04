# zju3dv.github.io/htd-refine（HTD-Refine 项目页）

> 来源归档（ingest）

- **标题：** HTD-Refine — CVPR 2026 Oral Award Candidate
- **类型：** site / project-page
- **官方入口：** <https://zju3dv.github.io/htd-refine/>
- **入库日期：** 2026-06-04
- **一句话说明：** 论文配套站点：强调用 **3D 速度 + 3D 加速度** 精炼现有 HMR，恢复 **全局坐标下自然人体运动**；含 Methods 三阶段说明、EMDB / RICH 对比视频与 **Jitter / MPJVE / MPJAE / WA-MPJPE** 等量化表。

## 页面公开信息（检索自 2026-06-04）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://zju3dv.github.io/htd-refine/> |
| arXiv | <https://arxiv.org/abs/2605.26879> |
| PDF | <https://arxiv.org/pdf/2605.26879> |
| 代码 | Coming Soon |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **TL;DR：** 用估计的 **3D velocity + 3D acceleration** 精炼现有 HMR，在 **global coordinates** 恢复 natural human motion。
2. **三阶段 Methods：** (a) 现成 HMR + 相机外参 → world 初始化；(b) **PVA-Net** 预测 stabilized 2D keypoints + 相机系速度/加速度；(c) 全序列优化 pose / 全局朝向 / 平移，约束含 2D、jerk、初始化正则。
3. **Video Results：** Web Video + EMDB 数据集对比（需勾选视频源预览）。
4. **Main Results 指标：** MPJVE（Mean Per-Joint Velocity Error）、MPJAE（Mean Per-Joint Acceleration Error）、Jitter、FS、WA-MPJPE、W-MPJPE、RTE。
5. **EMDB-2（移动相机）亮点：** GVHMR+HTD-Refine WA-MPJPE **69.2** vs GVHMR **118.7**；TRAM+HTD-Refine Jitter **6.6** vs TRAM **25.1**。
6. **RICH（静态相机）亮点：** GVHMR+HTD-Refine Jitter **3.6** vs **13.0**；TRAM+HTD-Refine **4.2** vs **18.7**。

## 对 wiki 的映射

- [`wiki/entities/paper-htd-refine-monocular-hmr.md`](../../wiki/entities/paper-htd-refine-monocular-hmr.md) — 方法栈、实验指标与机器人上游链路定位
