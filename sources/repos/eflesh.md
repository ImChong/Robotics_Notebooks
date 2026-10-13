# eFlesh（notvenky/eFlesh）

> 来源归档（ingest）

- **类型：** repository / tactile sensor design tool
- **仓库：** <https://github.com/notvenky/eFlesh>
- **项目主页：** <https://e-flesh.com/>
- **论文：** <https://arxiv.org/abs/2506.09994>
- **组织 / 作者：** New York University；Venkatesh Pattabiraman 等
- **License：** MIT（仓库 About / LICENSE）
- **入库日期：** 2026-10-13
- **复核日期：** 2026-10-13
- **内容摘要：** 提供 cut-cell 微结构生成工具与 notebook，将输入的凸形 OBJ/STL 转为贴合几何的 3D 打印结构；用户可调单元尺寸、材料刚度及磁铁位置，并为 Hall 磁力计板预留安装位。

## 仓库入口

- `microstructure/microstructure_inflators/`：C++ 微结构生成器与构建脚本。
- `regular.ipynb`、`cut-cell.ipynb`：生成常规及 cut-cell 结构的 notebook。
- 项目 README 建议 Ubuntu 20.04/22.04/24.04；几何依赖含 CGAL、Eigen、SuiteSparse 等。
- `build.sh cpu_nodes=n` 可调整 CPU 并行度；README 默认构建会使用 12 个 CPU 节点。
- 完成网格生成后，用户还需按自己的磁铁尺寸添加磁铁口袋、为磁力计板预留槽位，并在 3D 打印时暂停嵌入磁铁。

## 复现边界

仓库提供传感器结构生成工具，不会自动提供适配所有机器人外形的成品硬件。应核对 OBJ/STL 凸形输入要求、打印分辨率、磁力计板尺寸及磁场串扰；项目主页另提供 STL 下载、磁力计购买与可视化入口。

## 对 wiki 的映射

- [eFlesh 实体页](../../wiki/entities/paper-eflesh-magnetic-tactile-sensor.md)
- [论文来源](../papers/eflesh_arxiv_2506_09994.md)
- [项目主页归档](../sites/e-flesh-com.md)
