# 输出轮式双臂人形机器人的 URDF 的基本流程与工具链是什么？

> 来源归档（blog / 工程教程；用户提供的 20 页 PDF 网页打印版）。此处保留原文的主题结构和主要论点，不复制全文。

- **标题：** 输出轮式双臂人形机器人的 URDF 的基本流程与工具链是什么？
- **作者：** Zane Zhang（Zane Hub）
- **类型：** blog / tutorial
- **原始链接：** https://mp.weixin.qq.com/s/OejKe1RXuiQAY4ASm87POA
- **发表日期：** 2026-11-14（PDF 页面署名）
- **入库日期：** 2026-11-14
- **提供材料：** 用户上传的《输出轮式双臂人形机器人的 URDF的基本流程与工具链是什么？.pdf》，共 20 页；正文主要位于第 1–19 页，第 20 页是结束页
- **核对范围：** 已逐页通读 PDF 第 1–20 页，覆盖十四章、流程表、示意树、代码示例、跨仿真器注意事项与交付清单；对关键技术歧义另查 ROS/URDF 与 ros2_control 官方文档
- **通读：** 已完成
- **一句话说明：** 用轮式底盘、升降躯干、双机械臂与末端执行器为例，串起 CAD 零位、URDF 单树拓扑、xacro 参数化、惯量/mesh、ROS 2 控制接口与 SDF/MJCF/USD 转换验收。
- **知识映射：** [轮式双臂人形 URDF 建模工程指南](../../wiki/queries/wheeled-dual-arm-humanoid-urdf-workflow.md)、[URDF 基础概念](../../wiki/concepts/urdf-robot-description.md)

## 原文结构与分段摘录（归纳）

| PDF 页码 | 原文章节 | 核心内容 |
|---|---|---|
| 1–2 | 一、对象与拓扑 | 差速/麦克纳姆/四舵轮底盘，升降立柱、腰、左右臂、夹爪、头部；URDF 是单父节点树，闭环要另作处理 |
| 2–3 | 二、七步流程 | 冻结坐标系 → 列拓扑 → xacro 骨架 → 惯量 → 视觉/碰撞 mesh → 校验 → 控制/仿真适配 |
| 3–5 | 三、坐标系与四、link/joint | map/odom/base_footprint/base_link；SI、右手系、关节零位；visual、collision、inertial、axis、limit |
| 6–8 | 五、轮式底盘 | 差速轮与轮距/半径、麦克纳姆解算约定、舵轮的转向与驱动关节、从动轮接触 |
| 9–10 | 六、躯干、双臂与末端 | prismatic 升降、双臂镜像、平行夹爪 mimic |
| 10–12 | 七、xacro | 模块化包结构、宏/参数、度与弧度、展开及 check_urdf |
| 12–14 | 八、质量属性 | 正定惯量、主惯量三角条件、基本体解析解、平行轴定理、实测对照 |
| 14–15 | 九、mesh | 视觉与碰撞分离、格式/路径/尺度、减面与凸分解 |
| 15–16 | 十、工具链 | xacro、check_urdf、urdf_to_graphiz、RViz、Pinocchio、MoveIt 2、Gazebo、MuJoCo、Isaac |
| 16–18 | 十一、跨仿真器 | URDF→SDF/MJCF/USD 时的固定关节、惯量、名称及几何等问题 |
| 18 | 十二、ros2_control | 同名 joint 的 command/state interface 和底盘/机械臂控制器选型 |
| 19 | 十三、十四、验收与协作 | 15 项交付检查、ROS 包版本、CI 静态检查与 TF 基线比对 |

## 原文七阶段主要产出

1. **坐标系冻结：** CAD 装配零位、安装偏置、关节轴向和 base_link 的定义表。
2. **拓扑表：** 父子 link/joint 清单，标清哪些闭环只能等效化。
3. **参数化骨架：** 无 mesh 的 xacro 主链，可展开并通过 check_urdf。
4. **质量属性：** 质量、CoM、惯性矩阵的 CAD/实测来源及正确单位。
5. **几何与材质：** visual mesh 与简化 collision 分开，验证路径、坐标轴和比例尺。
6. **模型校验：** 运动学树、零位、极限、碰撞、关节方向、总质量与质心。
7. **接口与适配：** ros2_control 的接口映射，以及 SDF/MJCF/USD 导入后的回归。

## 原文中的典型操作

~~~bash
xacro robot.urdf.xacro > /tmp/robot.urdf
check_urdf /tmp/robot.urdf
urdf_to_graphiz /tmp/robot.urdf
gz sdf -p /tmp/robot.urdf > /tmp/robot.sdf
# display.launch.py 为项目自有启动文件，不是通用自带命令
ros2 launch robot_description display.launch.py
~~~

## 阅读时应保留的边界

- 文中轮半径、质量、阻尼、mesh 三角面数及毫米级验收阈值是**示例或经验建议**，不能直接当作所有平台的工程标准。
- 原文 p.5 将惯性坐标系写成“必须与 link 轴平行、URDF 无法旋转惯性主轴”；实际 URDF 的 inertial/origin 允许 rpy，详见 [URDF link 元素规范镜像](https://mirrors.nju.edu.cn/ros-wiki/urdf%282f%29XML%282f%29link.html)。
- 原文 p.13 将 g·mm² → kg·m² 的差距描述为六个数量级；按单位定义应乘 **10⁻⁹**，而 kg·mm² → kg·m² 才是 **10⁻⁶**。
- 原文 p.17 关于 SDF fixed joint 合并的描述缺少可通过 preserveFixedJoint 保留的配置边界，见 [gz_ros2_control 官方文档](https://control.ros.org/rolling/doc/gz_ros2_control/doc/index.html)。
- 原文 p.10/p.18 的 mimic/ros2_control 及“ROS 2 不再需要 transmission”不应当解释为各个版本/硬件插件都没有传动映射能力；见 [ros2_control Joint Kinematics](https://control.ros.org/master/doc/ros2_control/hardware_interface/doc/joints_userdoc.html)。
- 原文 p.9 关于立柱“伸出时 CoM 变化无法在 URDF 表达”，应区分**刚体自身恒定的惯量**与**由多个 prismatic link 配置变化引起的整机 CoM 改变**。
- PDF 为转载式网页打印载体；**引用归属作者 Zane Zhang / Zane Hub**，不是独立论文，也不代表官方软件规范。

## 对 wiki 的映射

- [轮式双臂人形 URDF 建模工程指南](../../wiki/queries/wheeled-dual-arm-humanoid-urdf-workflow.md)：落地流程、Mermaid、工具链、校验表以及原文勘误。
- [URDF（统一机器人描述格式）](../../wiki/concepts/urdf-robot-description.md)：纳入流程入口与来源。
- [URDF 连杆惯量对照真机检查](../../wiki/queries/urdf-link-inertia-real-robot-check.md)：惯量辨识与真机检查延伸。
