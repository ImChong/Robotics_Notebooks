# apollo-lab-yale.github.io/apollo-resources（URDD 浏览器检视）

> 来源归档（ingest）

- **标题：** In Browser Visualization（GitHub Pages 部署页，服务 URDD 资产检视）
- **类型：** site / demo-host
- **官方入口：** <https://apollo-lab-yale.github.io/apollo-resources/>
- **数据与页面源码仓库：** <https://github.com/Apollo-Lab-Yale/apollo-resources>
- **入库日期：** 2026-05-17
- **一句话说明：** 纯浏览器 **Three.js** 演示：从同域 `./robots/<name>/` 下并行拉取 **chain / urdf / mesh / convex hull / convex decomposition / link_shapes** 等 **URDD 模块 JSON**，用 `apollo-three-engine` 的 `RobotFromPreprocessor` 与 **关节滑条 FK 可视化** 组合；机器人与环境列表通过 **GitHub Contents API** 读 `apollo-resources` 仓的 `robots/`、`environments/` 目录动态填充。

## 页面公开信息（检索自 2026-05-17）

| 资源 | URL |
|------|-----|
| 在线演示 | <https://apollo-lab-yale.github.io/apollo-resources/> |
| 静态资源与 URDD 目录 | `Apollo-Lab-Yale/apollo-resources` 仓库 `robots/`、`environments/` |
| 三端引擎（import map） | `https://cdn.jsdelivr.net/gh/Apollo-Lab-Yale/apollo-three-engine@main/js/` |

## 与论文主张的对应关系（归纳）

- 首页脚本显式 `fetch(.../chain_module/module.json)`、`mesh_modules/...`、`link_shapes_modules/...` 等路径，和 arXiv:2512.23135 文中 **「模块化 JSON/YAML + Web 检视」** 叙述一致，可作为 **URDD 消费侧接口** 的公开样例。
- 论文脚注中的 **APOLLO Toolbox 文档**（Notion）为模块全集与维护说明的权威入口；本站点不替代该文档。

## 对 wiki 的映射

- [`wiki/entities/paper-urdd-universal-robot-description-directory.md`](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)
- [`sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md`](../papers/urdd_beyond_urdf_arxiv_2512_23135.md)
