# IDEA-Research/Motion-X 官方仓库阅读档案

- **仓库：** `IDEA-Research/Motion-X`（GitHub 官方主仓）
- **地址：** <https://github.com/IDEA-Research/Motion-X>
- **项目主页：** <https://motion-x-dataset.github.io>
- **通读：** 已完成
- **核对范围：** `README.md`、`LICENSE`、`datasets/README.md` 与划分文件（`all.txt`、`train.txt`、`val.txt`、`test.txt`）、`mocap-dataset-process/`（`README.md`、`humanml.py`、`face_motion_augmentation.py`）、`non-mocap-dataset-process/`（`README.md`、`aist.py`、`Load_motion_data.md`）、`PoseText/PoseText_README.md`、`tomato_represenation/`（`README.md`、`raw_pose_processing.py`、`smplx2joints.py`、`motion_representation.py`、`motionx2humanml/transfer_to_body_only_humanml.py`）以及 `visualization/`（`Readme.md`、`render_world_space_motion.py`）
- **关联维基：** [Motion-X 实体页](../../wiki/entities/dataset-bfm-motion-x.md) · [Motion-X++ 实体页](../../wiki/entities/dataset-bfm-motion-xpp.md)

## 仓库结构与数据形态核对

1. **许可条款（`LICENSE` 与 `README.md`）：**
   - 仓内 `LICENSE` 为 IDEA 非商业科学研究许可（`License for Non-commercial Scientific Research Purposes`），明确禁止商业、色情、军事或监控用途；
   - **Mocap 子集不随 Motion-X 直接分发原动作**：`mocap-dataset-process/README.md` 规定 `AMASS`（含 HumanML3D 对应部分）、`GRAB`、`EgoBody` 需用户先到原官网同意各自许可下载原动作，仓库/授权链接仅提供文本标签、映射逻辑和基于 BAUM 的 `face_motion_data`；
   - **非 Mocap 子集不直接分发原始视频**：Motion-X 主体提供 30fps `SMPL-X 322` 参数 `.npy` 与文本标签；`non-mocap-dataset-process/Load_motion_data.md` 与 `README.md`（2024.04.25 / 2025.03.01 更新）记录 **Motion-X++** 在同仓更新了多模态目录（`mesh_recovery/local_motion/*.json`、`mesh_recovery/global_motion/*.json`、`keypoints/*.json` 及 TOMATO 格式 HuggingFace 镜像 `YuhongZhang/Motion-Xplusplus`）。

2. **关键数据表示：**
   - **`smplx_322`（`322` 维逐帧向量，见 `README.md` 与 `Load_motion_data.md`）：**
     - `motion[:, 0:3]`：`root_orient`（全局根旋转）
     - `motion[:, 3:66]`：`pose_body`（21 个身体关节轴角，63 维）
     - `motion[:, 66:156]`：`pose_hand`（左右手各 15 关节，30×3 = 90 维）
     - `motion[:, 156:159]`：`pose_jaw`（下颌旋转，3 维）
     - `motion[:, 159:209]`：`face_expr`（面部表情系数，50 维）
     - `motion[:, 209:309]`：`face_shape`（面部形状系数，100 维）
     - `motion[:, 309:312]`：`trans`（全局平移，3 维）
     - `motion[:, 312:322]`：`betas`（静态体型参数，10 维）
   - **`tomato_represenation`（52 关节 / 623 维，继承自 HumanTOMATO）：**
     - `raw_pose_processing.py` 调用 `smplx2joints.py`（53 原生 SMPL-X 关节去下颌得 52 关节坐标）+ `DataLoader` 批处理导出 `new_joints`（`(T, 52, 3)`）；
     - `motion_representation.py` 沿用 HumanML3D `Skeleton` 进行腿长比例归一化 `uniform_skeleton`、贴地与初帧朝 `+Z` 对齐，输出 **623 维** `new_joint_vecs`；
     - `motionx2humanml/transfer_to_body_only_humanml.py` 可从 623 维中裁剪出前 22 关节，还原为 HumanML3D 兼容的 **263 维** `vector_263`。

3. **标准划分文件（`datasets/`）：**
   - `datasets/all.txt` 共 **81,079** 行，对应官方划分 `train.txt`（64,867 行，约 80%）、`val.txt`（4,046 行，约 5%）、`test.txt`（12,167 行，约 15%）。

4. **实际预处理与可视化入口：**
   - Mocap 接入：`mocap-dataset-process/humanml.py`（重定位 AMASS `SMPL-X G` 并转 Z-up/Y-up 轴序）、`egobody.py`、`grab.py` → `face_motion_augmentation.py`（将帧数对齐的 `face_motion_data` 通过 `F.interpolate(..., mode='linear')` 写入第 `156:309` 维）。
   - 非 Mocap 后处理：`non-mocap-dataset-process/aist.py` 将 AIST `trans` 除以 `94.0` 并翻转 Z 轴；`visualization/render_camera_space_motion.py` 与 `render_world_space_motion.py` 使用 PyTorch3D 从 COCO JSON 读取相机与 `smplx_params` 渲染叠画视频。
   - `PoseText/PoseText_README.md`：`body_texts/*.json` 与 `hand_texts/*.json` 按帧索引 `{frame_idx: description}` 存储，注意文档示例代码第 104 行存在一处笔误（对 `hand_f` 仍打开 `body_text_name + '.json'`，工程复用时需改用 `hand_text_name`）。
