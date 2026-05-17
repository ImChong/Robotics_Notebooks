# robot_motion_editor（cyoahs，浏览器 URDF 轨迹与关键帧编辑器）

- **标题**: 机器人关键帧编辑器（Robot Motion Editor）
- **代码**: <https://github.com/cyoahs/robot_motion_editor>
- **在线演示**: <https://motion-editor.cyoahs.dev>（README 写明托管于 Cloudflare Pages）
- **类型**: repo（Web 前端工具；MIT）
- **首次入库**: 2026-05-17

## 一句话摘要

纯浏览器侧（Vite + Three.js + `urdf-loader`）的 **URDF 加载 + CSV 轨迹编辑 + 残差式关键帧** 工具：双视口对比、曲线编辑器（含贝塞尔插值）、工程保存/恢复、IndexedDB 存大 mesh；支持 Unitree CSV 与 Seed CSV 互转导出。

## 隐私与数据流

README 强调 **完全本地运行**：解析与编辑在浏览器完成，无服务端上传（与需本地 Flask 后端的 npz 工具链对照时这是清晰分界）。

## 功能要点（README 归纳）

- **双视口**：左原始轨迹、右编辑结果，相机同步。
- **轨迹 / 关键帧**：残差关键帧系统；关节与浮动基可编辑；Shift+多选曲线。
- **工程文件**：可保存 URDF、轨迹、关键帧与编辑历史并完整恢复；自动保存用 localStorage + IndexedDB 分层。
- **动力学可视化（运动学层面）**：实时 **CoM** 与 **支撑多边形** 投影；非完整物理仿真。
- **快捷工具**：基座「平移对齐」使编辑后最低点与原始轨迹对齐；右下角坐标轴 gizmo 切正交视角。
- **多语言**：中 / 英界面。

## 数据格式

- **导入 CSV**：Unitree 风格（base xyz + 四元数 xyzw + 关节弧度）或 Seed 风格（Frame + cm/degree），内部统一到 Unitree 表示。
- **导出**：可选择 Unitree / Seed CSV 与 FPS；不同 FPS 时自动插值重采样。

## 模块结构（README `src/`）

`main.js`（双视口）、`urdfLoader.js`、`trajectoryManager.js`、`trajectoryFormatConverter.js`、`jointController.js`、`baseController.js`、`curveEditor.js`、`comVisualizer.js`、`axisGizmo.js`、`timelineController.js`、`cookieManager.js`、`indexedDBManager.js`、`themeManager.js`、`i18n.js`。

## 对 Wiki 的映射

- **`wiki/entities/robot-motion-keyframe-editors.md`**：与 MuJoCo keyframe kit、Project Instinct npz 编辑器并列的 **示教后处理 / 关键帧** 工具选型入口。
- **`wiki/concepts/motion-retargeting-pipeline.md`**：重定向或跟踪产线之后的 **手工轨迹修整** 场景。
- **`wiki/entities/unitree.md`** / **`wiki/tasks/locomotion.md`**：Unitree CSV 互操作与足式日志编辑的工程参照。
