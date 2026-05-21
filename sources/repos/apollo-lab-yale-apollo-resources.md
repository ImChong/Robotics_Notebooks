# apollo-resources（Apollo-Lab-Yale）

> 来源归档

- **标题：** apollo-resources
- **类型：** repo（URDD 机器人/环境资产与 GitHub Pages 宿主）
- **代码：** <https://github.com/Apollo-Lab-Yale/apollo-resources>
- **浏览器入口：** <https://apollo-lab-yale.github.io/apollo-resources/>（站点归档见 [sources/sites/apollo-lab-yale-apollo-resources-github-io.md](../sites/apollo-lab-yale-apollo-resources-github-io.md)）
- **入库日期：** 2026-05-17
- **一句话说明：** 存放 **按机器人分目录的 URDD 模块树**（`robots/<name>/..._module/` 等）与 **environments/** 并行结构，供论文配套 **Web 检视器** 与同组织的 **Rust / Python** 工具链加载；是理解 URDD **磁盘布局与相对路径契约** 的权威样本库。
- **沉淀到 wiki：** [URDD 论文实体](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)

---

## 目录语义（来自公开 Pages 与 API 用法归纳）

1. **`robots/`**：每个子目录名对应下拉菜单一项；内含 `chain_module`、`urdf_module`、`mesh_modules/*`、`link_shapes_modules/*` 等子树（与 arXiv:2512.23135 模块划分一致，具体文件以仓库为准）。
2. **`environments/`**：与机器人同构的预处理目录，可选与机器人同屏加载。
3. **GitHub Pages**：`index.html` 通过 `fetch` 读取相对路径下的 `module.json`，依赖仓库同步发布到 `gh-pages` 或等价分支（以仓库 Settings 为准）。

---

## 对 wiki 的映射

- 与 [URDD 论文实体](../../wiki/entities/paper-urdd-universal-robot-description-directory.md) 互链；作为 **「资产从哪来、长什么样」** 的工程参照。
