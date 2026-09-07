# open_duck_mini_viewer

> 来源归档

- **标题：** Open Duck Mini Viewer
- **类型：** repo
- **来源：** Masato Kobayashi（`mertcookimg`）
- **链接：** https://github.com/mertcookimg/Open_Duck_Mini_Viewer
- **项目页：** https://mertcookimg.github.io/Open_Duck_Mini_Viewer/
- **Stars：** 62（2026-09-07，GitHub）
- **License：** Apache-2.0（仓内 CAD 资产同样按原 Apache-2.0 再分发）
- **入库日期：** 2026-09-07
- **一句话说明：** 浏览器内 Open Duck Mini V2 GUI：three.js + URDF 看模型、脚本步态/关键帧动作、关节滑条与喷漆，零 Python、零真机，GitHub Pages 静态部署。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-duck-mini-viewer.md`](../../wiki/entities/open-duck-mini-viewer.md)

---

## 开源核查（步骤 2.5）

**已开源。** 项目页即 GitHub Pages 产物，页脚/README 链回本仓；源码、URDF/STL、本地启动脚本均可运行。无训练权重、无数据集发布——本工具也不需要。

| 资源 | 状态 | URL |
|------|------|-----|
| 代码 | 已开源 Apache-2.0 | https://github.com/mertcookimg/Open_Duck_Mini_Viewer |
| 在线 demo | 已部署（`main` push 自动发） | https://mertcookimg.github.io/Open_Duck_Mini_Viewer/ |
| CAD 资产 | 再分发自官方 Hub | `public/assets/open_duck_mini_v2/`（`robot.urdf` + STL） |
| 策略 / 物理引擎 | **不包含** | 步态与遥测是仓内 `Robot` 脚本模型，不是 Playground ONNX，也不是 MuJoCo WASM |

交叉：项目页归档 [`../sites/open-duck-mini-viewer-github-io.md`](../sites/open-duck-mini-viewer-github-io.md)；官方四仓见 [`open_duck_mini.md`](open_duck_mini.md)。

---

## 为什么值得保留

- 把 Open Duck Mini **从「要打印/要 Pi」降到「打开浏览器」**，适合教学、选型预览、CAD 爆炸图检查。
- 与 [Robot Viewer](../../wiki/entities/robot-viewer.md) 对照：那边是多格式 + MuJoCo WASM 物理；这边是 **单机型产品 GUI**（摇杆、动作库、喷漆、E-stop）。
- 2026-05-09 被 [IEEE Spectrum Video Friday](https://spectrum.ieee.org/video-friday-robotic-hand-dexterity) 点名：「Open Duck Mini is an open-source version of Disney’s BDX droids, and you can play with it in your browser.」

---

## 功能（README）

- 虚拟摇杆或 `WASD` / `QE` 驱动
- 动作：`home` / `stand` / `bow` / `wave` / `headbang` / `dance`（关键帧库在 `src/robot/motions.ts`）
- 关节滑条改 pose（进入 override）
- 3D 点选喷漆 / 随机配色
- CAD 检查：explode、wireframe、X-ray
- 关节角 sparkline、E-stop、世界/机体/关节坐标轴
- 列宽可拖；窄屏改为上 → 3D → 下堆叠

---

## 技术栈与目录

| 层 | 实现 |
|----|------|
| 前端 | Vite + React + TypeScript + Tailwind；Node ≥ 18 |
| 渲染 | three.js + [`urdf-loader`](https://github.com/gkjohnson/urdf-loaders) |
| 机器人模型 | `src/robot/Robot.ts`：脚本步态 + 里程计积分 + 关键帧混合 + gaze 平滑 |
| 关节定义 | `src/robot/joints.ts`：限位与 home pose（对齐 Runtime / Playground） |
| 部署 | `.github/workflows/deploy.yml` → GitHub Pages |

本地：`./scripts/start-all.sh`（或 Windows `start-all.ps1`）→ `http://localhost:5173`。门禁：`npm run typecheck` / `format:check` / `build`。

---

## `Robot` 模型要点（不是物理）

- 行走：`WALK_SPEED_MPS = 0.25`、`TURN_RATE_DPS = 90`，位姿限制在 `ARENA_HALF_M = 1.8`
- `gaitAngle`：按髋/膝/踝/横滚/偏航对相位正弦调制；膝角只向屈曲侧加幅，避免过伸
- IMU / 电池 / CPU 温度为 **装饰性随机游走**，不是 BNO055 或电源采样
- 头/颈/天线若干关节 `in_urdf: false`：GUI 可编，URDF 网格不一定有对应关节

---

## 与官方四仓的关系

本仓 **不是** apirrone 官方第五仓。CAD 来自 [Open_Duck_Mini](open_duck_mini.md)；home pose / 关节限位参考 [Runtime](open_duck_mini_runtime.md) 与 [Playground](open_duck_playground.md)。要训策略或上机，仍走那三条官方仓。

---

## 与本仓库 wiki 的映射

| 主题 | wiki 页 |
|------|---------|
| 本工具 | `wiki/entities/open-duck-mini-viewer.md` |
| 整机生态 | `wiki/entities/open-duck-mini.md` |
| 对照：多格式 + 物理 | `wiki/entities/robot-viewer.md` |
| 任务 | `wiki/tasks/locomotion.md` |
