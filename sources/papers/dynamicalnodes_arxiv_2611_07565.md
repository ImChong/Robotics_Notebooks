# dynamicalnodes: Block Diagrams to ROS2 Graphs（arXiv:2611.07565）

> 来源归档（paper）

- **标题：** dynamicalnodes: Block Diagrams to ROS2 Graphs
- **类型：** paper
- **作者与机构（论文首页）：** Nehal Singh Mangat、Floris van Breugel（Department of Mechanical Engineering, University of Nevada, Reno）；Humberto Ramos（Department of Mechanical and Aerospace Engineering, University of Florida）
- **arXiv：** <https://arxiv.org/abs/2611.07565>（v1，eess.SY，2026-11-05）
- **PDF：** <https://arxiv.org/pdf/2611.07565v1>
- **代码：** <https://github.com/nehalsinghmangat/dynamicalnodes>
- **项目页 / 教程：** <https://nehalsinghmangat.github.io/dynamicalnodes/tutorials/0_cruise_control/cruise_control_with_dynamicalnodes.html>
- **入库日期：** 2026-11-09
- **一句话说明：** 把反馈系统的每个模块写成离散时间动力系统的 (f, h) 函数对，先在 Python for 循环里组合仿真，再由 ROSNode 自动生成 rclpy 文件，得到与框图一一对应的 ROS 2 节点图。
- **核对范围：** 用 pdftotext 读完 arXiv v1 全文（摘要、I 引言、II 背景、III 框架、IV 巡航控制示例、V 相关软件、VI 结论与局限、致谢、作者贡献、参考文献列表），含所有代码图（Fig. 4 到 Fig. 14、Fig. 16）与公式；论文无附录和补充材料。Fig. 3、6、9、10、15、17、18 为图像，只读了图注与正文描述，未逐像素检查曲线，因此「ROS 输出与 Python 仿真一致」只按论文文字描述记录。参考文献只读列表，未逐条回溯原文。项目页返回 HTTP 200 但仅核对到可访问，未逐段读教程笔记本；GitHub 仓库只读了 README.md 与 CLAUDE.md（静态阅读，未运行代码）。
- **通读：** 已完成

## 开源状态

- **已开源**（步骤 2.5，2026-11-09 核查）：GitHub 仓库 nehalsinghmangat/dynamicalnodes 可访问；仓库 CLAUDE.md 自述 v0.2.1、Python >=3.12、MIT 许可（仓库自述，未单独核对 LICENSE 文件）。
- 论文正文本身未写许可证和版本号，只给出教程笔记本链接。

## 核心摘录

1. **问题（I 节）：** 作者点出两个障碍：控制算法缺少统一的编程格式（MATLAB、Python、面向对象、函数式并存）；实时系统的异步或不规则数据流需要回调式编程，而控制课程通常不教这种写法。
2. **方法（II、III 节）：** 每个模块是一对 (f, h)，f 为演化函数 x_{k+1} = f(x_k, u_k, θ)，h 为输出函数 y_k = h(x_k, u_k, θ)。f 或 h 缺省时用占位符 `*`，对应 Python 的 None。离散时间框图与 ROS 图都是同质有向图，方块对应节点，复合箭头对应话题（III 节开头）。
3. **API（III-A）：** 全框架只有两个类和三个方法：`DynamicalSystem.step`、`ROSNode.step`、`ROSNode.write_ROSNode_to_rclpy`。
4. **时间处理（III-C）：** 节点发布频率 f_pub = 1/Δt；封装系统里若有 t_k 参数则自动绑定到墙钟时间。
5. **数据处理（III-D、III-E）：** 订阅数据进队列，默认长度 5；用户提供 ros2py 与 py2ros 转换函数；`sync_mode="any"` 每来一条消息就触发一步，`sync_mode="all"` 等所有订阅队列在 `stale_after` 新鲜窗口内都有消息才触发。输入被消费后下一步传 None，不会重复使用旧值。
6. **示例（IV 节）：** 巡航控制，四个模块：车辆速度模型加测速噪声、正弦参考发生器、PID 控制器、线性 Kalman 滤波器。仿真参数见 Fig. 14：dt = 1.0，t 从 0 到 60，A_ref = 30，ω_ref = π/60，KP/KI/KD = 500/30/10，m = 1500，b = 50，Q = 0.1，R = 2.0。写 ROS 时（Fig. 16）改为 10 Hz，PID 节点 `sync_mode="all"`、`stale_after` 0.2，Kalman 节点 `sync_mode="any"`、`stale_after` 0.15，并把 F、B 按 dt_ros = 0.1 重算，Q 乘 dt_ros/dt。
7. **结果（IV-F）：** 用 rqt_graph 看拓扑（Fig. 17），用 PlotJuggler 画实时信号（Fig. 18），论文称其与 Python 仿真（Fig. 15）信号吻合。全文没有数值指标、误差统计、延迟或抖动测量，也没有真机实验。
8. **相关软件（V 节）：** 与 AlgebraicDynamics.jl（同一组合式动力系统范式，无机器人软件集成）、python-control（仿真对象内部组合，与 ROS 图无一一映射）、ros2_control（C++ 插件，需自己写 QoS 与数据处理，参数走 YAML）对比。作者定位是原型验证，之后可改写为 C++ 或包进 ros2_control。
9. **局限（VI 节，作者自述）：** 框架尚不成熟，没有被多个团队压力测试；不处理节点计算时间过长的情形，没有机制丢弃过期结果或切换算法。后续设想：接 python-control、do-mpc、casadi。

## 与「具身智能小站」条目的核对

小站条目（2026-11-09 更新，扫读类）称「把控制器、被控对象和观测器写成状态更新与输出函数，先组合仿真，再生成 ROS2 节点与通信配置；论文用巡航控制展示工作流，适合原型开发，尚不是复杂系统的大规模验证」。

- 与论文第 III、IV、VI 节一致。
- 「通信配置」论文原文写作 QoS 代码、话题、发布频率等元数据由 write_ROSNode_to_rclpy 写入 rclpy 文件（摘要、III-C），没有单独的配置文件。
- 「尚不是大规模验证」是论文自述（未被多团队压力测试）加上示例仅为一个单变量巡航系统的合并读法。
- 小站未给出机构，论文首页为内华达大学里诺分校与佛罗里达大学。

## 对 wiki 的映射

- [paper-dynamicalnodes](../../wiki/entities/paper-dynamicalnodes.md)
