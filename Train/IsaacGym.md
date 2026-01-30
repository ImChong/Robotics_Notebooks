# 🎮 Isaac Gym 示例学习清单（完整版）

## 📊 全部 28 个示例文件

|序号|文件名|难度|学习阶段|核心内容|关键知识点|
|---|---|---|---|---|---|
|1|`maths.py`|⭐|基础入门|数学运算基础|Vec3/Quat/Transform 操作、NumPy 转换|
|2|`bouncing_ball.py`|⭐|基础入门|最简单的物理模拟|创建 Sim、地面、球体、主循环结构|
|3|`asset_info.py`|⭐⭐|基础入门|资产内省|加载 URDF/MJCF、查询 bodies/joints/DOFs|
|4|`dof_controls.py`|⭐⭐|控制交互|关节控制方法|POS/VEL/EFFORT 模式、stiffness/damping|
|5|`joint_monkey.py`|⭐⭐|控制交互|DOF 属性动画|DOF 属性、限位、类型、坐标轴可视化|
|6|`transforms.py`|⭐⭐|控制交互|坐标变换可视化|Transform 计算、AxesGeometry、draw_lines|
|7|`apply_forces.py`|⭐⭐⭐|控制交互|施加力/力矩|Tensor API、apply_rigid_body_force_tensors|
|8|`apply_forces_at_pos.py`|⭐⭐⭐|控制交互|在指定位置施加力|apply_rigid_body_force_at_pos_tensors|
|9|`1080_balls_of_solitude.py`|⭐⭐|进阶功能|碰撞过滤|collision_group、collision_filter、状态重置|
|10|`body_physics_props.py`|⭐⭐⭐|进阶功能|物理属性设置|质量、惯性、摩擦、恢复系数|
|11|`actor_scaling.py`|⭐⭐⭐|进阶功能|Actor 缩放|资产缩放、运行时缩放|
|12|`graphics.py`|⭐⭐⭐|图形渲染|图形操作全面|纹理加载/创建、相机传感器、RGB/深度图像|
|13|`graphics_materials.py`|⭐⭐|图形渲染|网格材质|use_mesh_materials、mesh_normal_mode|
|14|`test_graphics_up.py`|⭐⭐|图形渲染|坐标轴设置|UP_AXIS_Y vs UP_AXIS_Z|
|15|`multiple_camera_envs.py`|⭐⭐⭐|图形渲染|多相机多环境|多相机创建、视图矩阵、图像保存|
|16|`interop_torch.py`|⭐⭐⭐⭐|PyTorch集成|PyTorch 张量交互|GPU Pipeline、wrap_tensor、相机张量|
|17|`convex_decomposition.py`|⭐⭐⭐|高级仿真|凸分解|VHACD 参数、复杂网格碰撞|
|18|`large_mass_ratio.py`|⭐⭐⭐|高级仿真|大质量比稳定性|密度设置、solver 迭代次数|
|19|`domain_randomization.py`|⭐⭐⭐⭐|高级仿真|域随机化|颜色/纹理/光照/相机随机化|
|20|`terrain_creation.py`|⭐⭐⭐⭐|高级仿真|地形创建|高度场、各种地形类型、三角网格|
|21|`projectiles.py`|⭐⭐⭐⭐|高级仿真|投射物动态创建|键盘/鼠标事件、动态设置速度、碰撞过滤|
|22|`soft_body.py`|⭐⭐⭐⭐|高级仿真|软体物理|FleX、杨氏模量、泊松比、应力可视化|
|23|`spherical_joint.py`|⭐⭐⭐|高级仿真|球关节控制|球关节、四元数→指数坐标、目标姿态|
|24|`franka_attractor.py`|⭐⭐⭐⭐|机器人控制|吸引子控制|AttractorProperties、末端跟踪|
|25|`franka_osc.py`|⭐⭐⭐⭐⭐|机器人控制|操作空间控制|OSC 算法、雅可比矩阵|
|26|`franka_cube_ik_osc.py`|⭐⭐⭐⭐⭐|机器人控制|IK + OSC 抓取|逆运动学、夹爪控制、大规模并行|
|27|`franka_nut_bolt_ik_osc.py`|⭐⭐⭐⭐⭐|机器人控制|螺母螺栓装配|状态机、精密操控、SDF碰撞|
|28|`kuka_bin.py`|⭐⭐⭐⭐⭐|机器人控制|Kuka 机械臂抓取|多物体场景、复杂任务逻辑|

---

## 📚 按学习阶段分组

### 🟢 第一阶段：基础入门（3个）

|序号|文件名|建议学习时间|进度|
|---|---|---|---|
|1|`maths.py`|0.5h|1|
|2|`bouncing_ball.py`|1h|1|
|3|`asset_info.py`|1h|1|

### 🔵 第二阶段：控制与交互（5个）

|序号|文件名|建议学习时间|进度|
|---|---|---|---|
|4|`dof_controls.py`|1.5h|1|
|5|`joint_monkey.py`|1h|1|
|6|`transforms.py`|1h|1|
|7|`apply_forces.py`|1h|1|
|8|`apply_forces_at_pos.py`|0.5h|0.2|

### 🟡 第三阶段：进阶功能（4个）

|序号|文件名|建议学习时间|进度|
|---|---|---|---|
|9|`1080_balls_of_solitude.py`|1h||
|10|`body_physics_props.py`|1.5h||
|11|`actor_scaling.py`|0.5h||
|12|`interop_torch.py`|2h||

### 🟠 第四阶段：图形渲染（4个）

|序号|文件名|建议学习时间|进度|
|---|---|---|---|
|13|`graphics.py`|2h||
|14|`graphics_materials.py`|0.5h||
|15|`test_graphics_up.py`|0.5h||
|16|`multiple_camera_envs.py`|1h||

### 🔴 第五阶段：高级仿真（7个）

|序号|文件名|建议学习时间|进度|
|---|---|---|---|
|17|`convex_decomposition.py`|1h||
|18|`large_mass_ratio.py`|1h||
|19|`domain_randomization.py`|2h||
|20|`terrain_creation.py`|2h||
|21|`projectiles.py`|1.5h||
|22|`soft_body.py`|2h||
|23|`spherical_joint.py`|1h||

### ⚫ 第六阶段：机器人控制（5个）

|序号|文件名|建议学习时间|进度|
|---|---|---|---|
|24|`franka_attractor.py`|2h||
|25|`franka_osc.py`|3h||
|26|`franka_cube_ik_osc.py`|4h||
|27|`franka_nut_bolt_ik_osc.py`|4h||
|28|`kuka_bin.py`|3h||

---

## 🏃 快速运行命令

```bash
cd python/examples
python <文件名> --physics_engine=physx
```

**总计：28 个示例文件，预计完整学习时间：40-50 小时** 🎯