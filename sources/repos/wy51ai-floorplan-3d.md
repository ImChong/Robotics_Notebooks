# floorplan-3d（wy51ai/floorplan-3d）

> 来源归档

- **标题：** 户型装修设计（floorplan-3d）— 纯前端 2D/3D 户型装修设计工具
- **类型：** repo（单文件前端应用 + GitHub Pages 在线 Demo）
- **作者：** wy51ai
- **链接：** https://github.com/wy51ai/floorplan-3d
- **在线体验：** https://wy51ai.github.io/floorplan-3d/
- **入库日期：** 2026-10-14
- **一句话说明：** 在 2D 平面图上摆放家具、拆改墙体、测量尺寸，并一键切换到 Three.js 3D 场景鸟瞰或第一人称漫游；整个应用是一个 `index.html`，无需构建。
- **开源状态：** **已开源** — MIT。
- **沉淀到 wiki：** 是 → [`wiki/entities/floorplan-3d-wy51ai.md`](../../wiki/entities/floorplan-3d-wy51ai.md)

## README 要点（归纳）

- **2D 平面布置：** 按原始户型 1:60 / 1:100 比例显示，单位 mm；从家具库拖入 60 余种家具家电，支持移动、旋转、改尺寸与贴墙吸附；测量工具；拆改非承重墙，承重墙单独标示；尺寸标注 / 房间名 / 家具 / 网格 / 承重墙图层开关。
- **3D 场景：** 鸟瞰、斜视、俯视与房间定位；桌面端 WASD + 鼠标漫游，触屏虚拟摇杆，可点门开关；全高墙 / 剖切墙切换、日照时间滑块、夜景灯光；3D 中选中、拖动家具与 2D 方案实时同步。
- **方案与统计：** 房间面积与套内使用面积自动统计；逐房间更换地面材料，按面积加 5% 损耗估算造价；撤销 / 重做，浏览器本地自动保存；中英文界面；导出 PNG，导入 / 导出方案 JSON。
- **快速开始：** 克隆后直接用浏览器打开 `index.html`，或 `python3 -m http.server 8000`；Three.js 通过 jsDelivr CDN 加载，首次进入 3D 场景需联网。
- **快捷键：** `T` 切换 2D/3D；`V`/`M`/`X` 选择 / 测量 / 拆墙；`R`/`Shift+R` 旋转；`Ctrl/⌘+Z` 撤销等。
- **自定义户型：** 户型几何、家具清单与材料配置以 `ROOMS`、`WALLS`、`WINS`、`MATS`、`LIB` 等数据定义在 `index.html` 中，换户型需改这些数据，不是上传户型图自动识别。

## 为何值得保留

- 浏览器 Three.js 交互应用样本：与 [img2threejs](img2threejs.md) 等 Web 3D 条目同属可视化 / 交互资产侧，非机器人仿真资产。
