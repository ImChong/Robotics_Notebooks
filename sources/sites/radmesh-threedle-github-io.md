# RADmesh Project Page（Threedle Lab）

> 来源归档

- **标题：** RADmesh: Remesh-Aware Mesh Deformation
- **类型：** site / project page
- **URL：** <https://threedle.github.io/radmesh/>
- **论文：** <https://arxiv.org/abs/2608.17182>
- **代码：** <https://github.com/threedle/radmesh>
- **机构：** University of Chicago（Threedle Lab）· USC · Technion
- **入库日期：** 2026-08-20
- **一句话说明：** ECCV 2026 **Oral** 官方页：remesh-in-the-loop 文本引导网格形变、Gallery、方法概览、UV 工作流与 BibTeX。

## 开源状态（项目页核查，2026-08-20）

| 项 | 状态 |
|----|------|
| Paper | arXiv **2608.17182** |
| Code | **GitHub 可点** → `threedle/radmesh` |
| 结论 | **已开源** — 项目页 Code 与仓库 README 一致 |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| Hero | 文本 prompt → 选区形变 + remesh；Spot 牛等多 prompt teaser |
| Gallery | 局部生长 + 全局 detailization |
| Overview | Q 优化 → 渲染 CSD → 每 N epoch remesh + 状态插值 |
| Deformation | 6D Q；dARAP Local/Global；scale + rotation |
| Remeshing | Botsch–Kobbelt；粗到细 target length；barycentric 属性插值 |
| Applications | 多 prompt 迭代；UV/纹理区外不变 |
| BibTeX | `dinh2026radmesh` |

## 对 wiki 的映射

- 论文：[`sources/papers/radmesh_arxiv_2608_17182.md`](../papers/radmesh_arxiv_2608_17182.md)
- 代码：[`sources/repos/radmesh.md`](../repos/radmesh.md)
- 沉淀 **[`wiki/entities/paper-radmesh.md`](../../wiki/entities/paper-radmesh.md)**
