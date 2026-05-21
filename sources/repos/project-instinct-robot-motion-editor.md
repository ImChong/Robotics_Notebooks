# robot-motion-editor（project-instinct，URDF + NPZ 浏览器编辑器）

- **标题**: Robot Motion Editor
- **代码**: <https://github.com/project-instinct/robot-motion-editor>
- **维护方**: README 写明由 **Project Instinct group** 开发
- **类型**: repo（Flask 后端 + 浏览器 Three.js；仓库未在 README 中单列 License 字段，以仓库根 `LICENSE` 文件为准）
- **首次入库**: 2026-05-17

## 一句话摘要

**Web 3D（Three.js + urdf-loader）+ 自研 2D 曲线编辑器**：加载 URDF 与 `.npz` 运动（关节角 + 基座位姿），逐通道拖拽关键帧、时间轴 scrub、可选 **滑动平均平滑**，再导出 NPZ；**解析与写回 npz 依赖本地 Flask**（与纯静态 cyoahs 编辑器对照）。

## 技术栈

- 前端：HTML5、ES module JS；Three.js；`urdf-loader`。
- 后端：`app.py`（Flask）+ `numpy`；用于 **解析与保存** NPZ。

## 快速开始（README）

1. `pip install flask numpy`
2. 将机器人 URDF 与 mesh 置于 `static/`（路径与 URDF 内 `package://` 等引用需一致）。
3. `python app.py` → 浏览器打开 `http://127.0.0.1:5000`。

## NPZ 约定（README）

期望数组键：

- `joint_pos`: `(T, n_joints)`
- `base_pos_w`: `(T, 3)`
- `base_quat_w`: `(T, 4)`，**wxyz** 顺序
- `joint_names`: 与 URDF 关节名对齐的字符串列表
- `framerate`：可选，缺省导出按 30 FPS

## 对 Wiki 的映射

- **`wiki/entities/robot-motion-keyframe-editors.md`**：与同源研究门户并列的 **NPZ 曲线编辑** 工具说明。
- **`wiki/entities/project-instinct.md`**：同一组织的公开工具链交叉引用。
- **`wiki/methods/imitation-learning.md`** / **`wiki/tasks/manipulation.md`**：演示数据后处理与关键帧修正场景。
