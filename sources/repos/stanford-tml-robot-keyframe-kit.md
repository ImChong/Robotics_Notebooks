# robot_keyframe_kit（Stanford-TML，MuJoCo 关键帧与轨迹编辑器）

- **标题**: robot-keyframe-kit
- **代码**: <https://github.com/Stanford-TML/robot_keyframe_kit>
- **PyPI**: `pip install robot-keyframe-kit`；CLI 入口 `keyframe-editor`
- **类型**: repo（Python + Viser + MuJoCo；MIT）
- **机构**: Stanford TML（README 归属与引用文献一致）
- **首次入库**: 2026-05-17

## 一句话摘要

面向 **任意 MuJoCo 兼容 MJCF** 的 **可视化关键帧编辑器**：自动机构检测（差速、齿轮、并联）、镜像模式、地面贴合、末端 site 跟踪、**Mink IK** QP 求解；可加载 `scene.xml`（推荐）或仅为 `robot.xml` 时自动补地面；运动保存为 **LZ4 压缩 joblib pickle**。

## 安装与入口

- **用户**: `pip install robot-keyframe-kit` → `keyframe-editor <xml_path> [--name ...]`。
- **开发**: 可编辑安装 `pip install -e ".[dev]"`；可选 `bash scripts/setup_assets.sh` 拉 Menagerie 资产做示例。

## 功能要点（README Features）

- 视觉关节控制；物理步进测试关键帧与轨迹；根体贴地基于最低碰撞几何；末端执行器 site / body 自动检测与跟踪。
- **配置 YAML**：`root_body`、`end_effector_sites`、`mirror_pairs` / `mirror_signs`、`dt`、`save_dir`、`scene.auto_inject_floor` 等；支持 `--generate-config`。

## 保存文件结构（README 表格摘要）

`joblib.load` 解压后典型键：`keyframes`（列表字典，含 `name`、`motor_pos`、`joint_pos`、`qpos`）、`timed_sequence`（`(name, duration)`）、`time`、`qpos` 轨迹、`motor_vel` / `joint_vel`、`action`（可选）、各 body / site 位姿速度、`is_robot_relative_frame` 等。

## 学术引用（README BibTeX）

工具 README 请引用：

- Yang et al., *Locomotion Beyond Feet* — arXiv:2601.03607（2026-01）。
- Shi et al., *ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation* — arXiv:2502.00893（2025）。

## 对 Wiki 的映射

- **`wiki/entities/robot-motion-keyframe-editors.md`**：MuJoCo 原生工作流上的关键帧 / 示教数据编辑代表实现。
- **`wiki/entities/mujoco.md`**：MJCF 资产消费侧工具链补充。
- **`wiki/concepts/motion-retargeting-pipeline.md`**：仿真内参考姿态编排与轨迹导出接口形态参照。
