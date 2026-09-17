# DIJIT Binocular Robotic Head（GitLab · 标定与控制）

> 来源归档

- **标题：** dijit-binocular-robotic-head
- **类型：** repo / hardware / calibration / active vision
- **链接：** <https://gitlab.nvision.eecs.yorku.ca/robots/dijit-binocular-robotic-head>
- **机构：** 约克大学（York University）NVision / Tsotsos Lab
- **硬件论文：** [DIJIT arXiv:2512.07998](../papers/humanoid_pnb_dijit-a-robotic-head-for-an-active-observer.md)
- **立体算法：** [CBS Zenodo](../repos/convergent_binocular_stereo_zenodo.md) — README 要求 clone 本仓获取 **motor–camera calibration**
- **入库日期：** 2026-09-17
- **一句话说明：** DIJIT 主动双目机器人头的 **GitLab 工程仓**（非 GitHub）：电机–相机标定与 CBS 会聚立体复现的硬件侧依赖；与 [Science Robotics CBS 论文](../papers/convergent_binocular_stereo_scirobotics_2026.md) 同课题组。
- **开源状态：** **已开源**（GitLab；CBS Zenodo 文档显式引用）

---

## 与 CBS / DIJIT 论文关系

| 层级 | 内容 |
|------|------|
| 硬件 + 扫视 | DIJIT 论文（arXiv:2512.07998）：9 机械 + 4 光学 DoF，vergence/version/cyclotorsion |
| 深度算法 | CBS（Sci. Robot. eaec7205）：会聚几何下 Gabor 粗到细视差 |
| 本仓 | 标定与控制代码，支撑 CBS-BM 采集与 `run_one.py` 电机参数 |

---

## 交叉链接

- Wiki：[paper-notebook-dijit-a-robotic-head-for-an-active-observer.md](../../wiki/entities/paper-notebook-dijit-a-robotic-head-for-an-active-observer.md)
- Wiki：[paper-convergent-binocular-stereo.md](../../wiki/entities/paper-convergent-binocular-stereo.md)
- Zenodo CBS：[convergent_binocular_stereo_zenodo.md](convergent_binocular_stereo_zenodo.md)
