# C/C++ 运动控制开源库分类整理

> 来源归档（blog / 微信公众号）

- **标题：** C/C++ 运动控制开源库分类整理
- **类型：** blog
- **作者：** Qt历险记（微信公众号，「qt」合集）
- **原始链接：** https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw
- **发表日期：** 2026-11-02
- **入库日期：** 2026-11-03
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 2340 字符
- **一句话说明：** 把 C/C++ 运动控制开源库分成轨迹规划、机器人动力学 / 整机控制、工业总线与控制栈、ROS / ROS2 运动控制栈四类（文章开头称五类：轨迹生成、机器人动力学 / 控制、工业 PLCopen 运控、嵌入式电机底层、ROS 生态；正文实际按四个小节排列，加一节选型建议），标注语言、许可与适用场景。
- **说明：** 文章是转述式清单，**没有给出处**。本次对每个项目回到一手资料核对（仓库 LICENSE / README、官方文档、论文），并在各节点标出与文章不一致的地方。文章中的「WBC Whole-Body Control」只给了通用名称、没有仓库，不能对应到唯一项目，因此只链接概念页 [whole-body-control](../../wiki/concepts/whole-body-control.md)，不建节点。

## 项目 → 节点与一手核对（14 个有仓库 / 官方文档的项目）

| 项目 | 类别 | 节点 | 状态 | 与文章的主要差异 |
|------|------|------|------|------------------|
| Pinocchio | 机器人动力学 / 整机控制库 | [pinocchio](../../wiki/entities/pinocchio.md) | 复用 | 一致。Crocoddyl README 确实说明用 Pinocchio 计算动力学与解析导数。 |
| Crocoddyl | 机器人动力学 / 整机控制库 | [crocoddyl](../../wiki/entities/crocoddyl.md) | 复用 | 一致。README 描述为接触序列下的最优控制库，求解器基于 DDP，并用 Pinocchio 计算动力学与解析导数。 |
| SOEM | 工业总线与控制栈 | [soem](../../wiki/entities/soem.md) | 复用 | 文章只写「GPL」，漏了关键约束：LICENSE.md 明确这是 GPLv3 + 商业双许可，用于商业产品多半需要购买商业许可。 |
| IgH EtherCAT Master | 工业总线与控制栈 | [igh-ethercat-master](../../wiki/entities/igh-ethercat-master.md) | 复用 | 仓库同时含 COPYING（GPLv2）与 COPYING.LESSER（LGPL v2.1），文章只写 GPL；硬实时依赖内核补丁来自文章，未在许可文件中核对。 |
| SimpleFOC（Arduino-FOC） | 工业总线与控制栈 | [simplefoc](../../wiki/entities/simplefoc.md) | 复用 | 许可一致；平台列表未逐项核对。 |
| ros2_control | ROS / ROS2 运动控制栈 | [ros2-control](../../wiki/entities/ros2-control.md) | 复用 | 文章未写许可；仓库 LICENSE 为 Apache-2.0。 |
| MoveIt 2 | ROS / ROS2 运动控制栈 | [moveit2](../../wiki/entities/moveit2.md) | 复用 | 许可一致。 |
| Ruckig | 轨迹规划库 | [ruckig](../../wiki/entities/ruckig.md) | 新建 | **许可不一致**：文章写社区版 MPL-2.0；仓库 LICENSE 与 README 徽章均为 **MIT**。 |
| pyplcopen / plcopen 运动控制内核 | 轨迹规划库 | [plcopen-pyplcopen](../../wiki/entities/plcopen-pyplcopen.md) | 新建 | **名称对应**：文章写的「pyplcopen」是 PyPI 包名，真实源码仓库是 **lusipad/plcopen**（PyPI 页面 Repository 链接指向它）；Python 侧是仿真门面，内核是 C++17。 |
| FastAccelStepper | 轨迹规划库 | [fastaccelstepper](../../wiki/entities/fastaccelstepper.md) | 新建 | **「S 型加减速」「无浮点」未核对**：README 的已读部分描述的是高速步进与线性加减速平台表，没有读到 S 曲线或「无浮点」字样；这两点按文章表述保留，未核实。 |
| Robotics Library（RL） | 机器人动力学 / 整机控制库 | [robotics-library-rl](../../wiki/entities/robotics-library-rl.md) | 新建 | 许可：README 写明为 **BSD 2-clause**（文章只写「BSD」）。 |
| Control Toolbox（ct） | 机器人动力学 / 整机控制库 | [control-toolbox-ct](../../wiki/entities/control-toolbox-ct.md) | 新建 | **维护状态**：README 顶部声明（2021 年 7 月）该库「currently only scarcely maintained」，对 bug 与需求响应缓慢；仓库最近提交 2021-07-05。文章未提，选型需注意。 |
| RTmotion（Intel ECI） | 工业总线与控制栈 | [rtmotion-intel-eci](../../wiki/entities/rtmotion-intel-eci.md) | 新建 | **未能定位公开 GitHub 仓库**：我尝试了常见仓库路径与搜索，没有找到可访问的 RTmotion 源码仓库；一手资料只有 Intel ECI 官方文档，因此**许可与最近提交均未核实**，本页不编造仓库地址。 |
| OMPL（Open Motion Planning Library） | ROS / ROS2 运动控制栈 | [ompl](../../wiki/entities/ompl.md) | 新建 | 许可：LICENSE 写 3-clause BSD，与文章一致。 |

## 选型建议（文章）与核对

| 场景 | 文章建议 | 核对 |
|------|----------|------|
| 自研 EtherCAT 工业运动控制器（CNC / 机械手） | pyplcopen（plcopen 内核）+ SOEM / IgH + Ruckig（可选） | plcopen README 写 EtherCAT 未开工，需自行接主站；SOEM 商用需商业许可；Ruckig 社区版中间路点非实时 |
| 只需多轴平滑 S 曲线、jerk 限制轨迹 | Ruckig | 许可为 MIT（非文章所写 MPL-2.0）；带中间路点的实时计算需 Pro |
| 机械臂动力学、力矩控制、DDP 优化 | Pinocchio + Crocoddyl | 一致 |
| 单片机无刷 FOC 电机环 | SimpleFOC | 一致 |
| ROS2 机械臂项目 | MoveIt2 + ros2_control + Ruckig | 一致；ros2_control 为 Apache-2.0 |
| 步进电机单片机 | FastAccelStepper | S 型加减速、无浮点两点未核对 |

## 对 wiki 的映射

- 地图：[C/C++ 运动控制开源库地图：轨迹生成、动力学、工业总线与 ROS 栈（一手资料核对版）](../../wiki/overview/cpp-motion-control-open-source-libraries-map.md)
