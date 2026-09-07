# Open Duck Mini Viewer（GitHub Pages）

> 来源归档

- **标题：** Open Duck Mini Viewer
- **类型：** site（GitHub Pages 静态 SPA）
- **来源：** Masato Kobayashi（`mertcookimg`）
- **链接：** https://mertcookimg.github.io/Open_Duck_Mini_Viewer/
- **仓库：** https://github.com/mertcookimg/Open_Duck_Mini_Viewer → [`../repos/open_duck_mini_viewer.md`](../repos/open_duck_mini_viewer.md)
- **入库日期：** 2026-09-07
- **一句话说明：** Open Duck Mini Viewer 的官方在线 demo：纯前端 URDF 查看 + 脚本步态/动作库，`main` 每次 push 由 GitHub Actions 自动发布。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-duck-mini-viewer.md`](../../wiki/entities/open-duck-mini-viewer.md)

---

## 开源核查（步骤 2.5）

2026-09-07 打开项目页：页面即 Viewer 本体（Vite/React SPA，无独立「Code will be released」文案）。页内可交互 3D 鸭；源码入口在 GitHub README / 仓库 homepage 字段互指。

**结论：已开源 + 已部署。** 代码与 Pages 同源；无权重或数据集。

- **代码：** https://github.com/mertcookimg/Open_Duck_Mini_Viewer
- **在线 demo：** 本页 URL

---

## 页面能做什么（与 README 对齐）

站点是 **单一应用**，不是文档站。打开即可：

- 摇杆 / `WASD` `QE` 走动
- 触发 `home` / `stand` / `bow` / `wave` / `headbang` / `dance`
- 滑条改关节、点选喷漆、爆炸图 / 线框 / X-ray
- 关节 sparkline、E-stop、坐标轴开关

步态、IMU、电量由浏览器内 `Robot` 模型生成，因此静态托管即可跑通。

---

## 部署与隐私

- 工作流：仓库 `.github/workflows/deploy.yml`，合入 `main` 后发布；**PR 构建不自动上线**。
- 官方 Pages 开了 **Google Analytics 4**（聚合访问）；本地 `npm run dev` 与 fork 部署不向该 GA4 属性送数。

---

## 抓取注意

页面为客户端 SPA，无正文 Markdown。自动化抓取只能得到壳标题；功能清单以仓库 README 与 `src/` 为准。2026-09-07 入库时对照 README + `Robot.ts` / `motions.ts` / `joints.ts`，并核过 IEEE Spectrum Video Friday 条目。

---

## 对 wiki 的映射

| 主题 | wiki 页 |
|------|---------|
| Viewer 实体 | `wiki/entities/open-duck-mini-viewer.md` |
| 官方整机 | `wiki/entities/open-duck-mini.md` |
| 通用 Web 查看器对照 | `wiki/entities/robot-viewer.md` |
