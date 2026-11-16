# amazon-far/holosoma OmniRetarget 重定向与数据集卡精读归档

- **类型：** 开源仓库与数据集说明精读
- **关联仓库：** <https://github.com/amazon-far/holosoma>（`src/holosoma_retargeting/` 与 `demo_scripts/`）
- **关联数据集：** <https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset>
- **通读：** 已完成
- **核对范围：** `OmniRetarget_Dataset` HuggingFace 数据集卡（数据子集、规模、`qpos` 维度、MIT 许可与 `visualize.py`）以及 `amazon-far/holosoma` 仓库中的 OmniRetarget 重定向与跟踪联动链路（`src/holosoma_retargeting/README.md`、`holosoma_retargeting/examples/robot_retarget.py`、`holosoma_retargeting/src/interaction_mesh_retargeter.py`、`holosoma_retargeting/data_conversion/convert_data_format_mj.py`、`demo_scripts/demo_lafan_wb_tracking.sh`、`demo_scripts/demo_omomo_wb_tracking.sh`）；论文全文归档见 [omniretarget_arxiv_2509_26633.md](../papers/omniretarget_arxiv_2509_26633.md)。

---

## 一、HuggingFace 数据集卡核对（`omniretarget/OmniRetarget_Dataset`）

1. **许可与开源范围**：数据集卡标注整体采用 **MIT** 许可；因 Ubisoft LAFAN1 原始数据许可限制，LAFAN1 重定向轨迹未随 HF 仓库分发，需用户利用 `holosoma` 仓库中的重定向代码自行生成。
2. **三个公开子集（合计 4.0 小时）**：
   - `robot-object/`（3.0 h，源自 OMOMO，Unitree G1 搬运物体）
   - `robot-terrain/`（0.5 h，自采 MoCap，复杂地形攀爬/跨障）
   - `robot-object-terrain/`（0.5 h，自采 MoCap，人–物–地形联合交互）
   - 另附 `models/` 目录（URDF / SDF / OBJ），仅供 `visualize.py`（基于 Drake + Meshcat）渲染检查使用，训练加载轨迹本身不需要。
3. **文件与张量 Schema**：每条 `.npz` 轨迹含标量 `fps`（如 `30.0`）与 `qpos`（`[T, D]`），其中机器人广义位姿固定 **36 维**（浮动基四元数和平移 `[qw, qx, qy, qz, x, y, z]` 7 维 + Unitree G1 29 关节角），无物体任务 `D=36`，含交互物体时追加物体 7 维位姿 `[qw, qx, qy, qz, x, y, z]` 即 **`D=43`**。

---

## 二、holosoma_retargeting 源码核对（`amazon-far/holosoma`）

1. **核心入口与 SQP 重定向器**：
   - `holosoma_retargeting/examples/robot_retarget.py` 通过 `tyro.cli(RetargetingConfig)` 解析任务类型（`robot_only` / `object_interaction` / `climbing`）与数据格式（`smplh` / `mocap` / `lafan`），加载动作数据后构建 `InteractionMeshRetargeter`。
   - `InteractionMeshRetargeter`（`src/interaction_mesh_retargeter.py`）使用 `cvxpy` 以 SQP 方式迭代求解逆运动学：目标函数最小化交互网格拉普拉斯形变（保持人体/物体/地形相对拓扑关系）与名义姿态跟踪，硬约束包含地面与物体非穿透（`activate_obj_non_penetration`）、足端防滑锁（`activate_foot_sticking` + `foot_sticking_tolerance`）、关节限位与信赖域 `step_size`，并支持 Viser 实时调试预览。
2. **LAFAN1 自主重定向与 WBT 全流程联动（`demo_scripts/`）**：
   - `demo_lafan_wb_tracking.sh` 展示完整自生成闭环：下载 `lafan1.zip` BVH 后调用 `data_utils/extract_global_positions.py` 提取世界系关节位置 `.npy` → 调用 `examples/robot_retarget.py --data_format lafan --retargeter.foot-sticking-tolerance 0.02` 输出 G1 `qpos` `.npz` → 调用 `data_conversion/convert_data_format_mj.py --output_fps 50` 结合 MuJoCo 模型重采样并补全全链路运动学参考（`*_mj_fps50.npz`）→ 调用 `src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt` 启动全身跟踪 RL 训练。
   - `demo_omomo_wb_tracking.sh` 对 `demo_data/OMOMO_new`（InterMimic 预处理 `.pt` 格式）执行完全对称的重定向 → 50 Hz MuJoCo 转换 → IsaacLab `exp:g1-29dof-wbt` 训练链路。
