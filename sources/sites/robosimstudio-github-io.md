# RoboSimStudio 项目主页

- **类型：** 项目页
- **入口：** <https://robosimstudio.github.io/>
- **代码：** <https://github.com/robosimstudio/robosimstudio>（Apache-2.0，已公开）
- **技术报告：** <https://arxiv.org/abs/2610.18835>
- **团队：** UC Berkeley AUTOLab 与 NVIDIA
- **核查日期：** 2026-10-16
- **收录日期：** 2026-10-15
- **一句话说明：** 面向机器人操作任务 Real2Sim 的开放交互式 GPU 仿真工具包，可由人或编码 Agent 根据手机视频构建并迭代检查物理场景。

## 项目定位与演示

项目页将 RoboSimStudio 定位为可加速机器人操作任务 Real2Sim 的交互式物理仿真工具。流程示例以一段手机拍摄的双臂海绵擦拭视频和一行提示为输入，由 Claude 或 Codex 等编码 Agent 搭建场景、相机与物理模型，约一小时内得到可交互的仿真。官方页面展示的这一案例对应论文报告的 56 分钟，不是对任意任务耗时的保证。

仿真采用多物理组件：刚体在 Newton 中运行，海绵的软体形变由 SuperDex FEM 模拟；项目仓库还列出 Warp 与 MuJoCo Warp 等底层组件。人可以在交互式 Studio 中检查和调整物理参数，Agent 则根据仿真反馈继续迭代。

## 结果范围

论文展示了 18 项接触丰富操作任务仿真，覆盖刚体、关节机构、线缆、布料及软体对象。代表性双臂擦拭案例中，Graph-as-Policy 在仿真反馈下迭代后部署到真机并完成擦拭；真机运行也发现并促使修正了支撑杆约束与实物不符的问题。这是单项案例的闭环演示，不能直接外推为广泛任务上的 Sim2Real 成功率。

## 开源状态核查

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 官方仓库公开，许可证为 Apache-2.0 |
| **项目页** | <https://robosimstudio.github.io/> |
| **技术报告** | <https://arxiv.org/abs/2610.18835> |
| **基础组件** | Newton、Warp、MuJoCo Warp；海绵软体使用 SuperDex FEM |
| **复现入口** | 仓库提供 examples/；README 给出 Agent 根据本地视频与任务描述创建环境的提示 |

项目页可作为演示与概览入口；实现细节、许可证和运行入口以[官方仓库](../repos/robosimstudio.md)及其 README 为准。

## 对 wiki 的映射

- 工具实体页：[wiki/entities/robosimstudio.md](../../wiki/entities/robosimstudio.md)
- Newton 基础引擎：[wiki/entities/newton-physics.md](../../wiki/entities/newton-physics.md)

## 参考链接

- 项目主页：<https://robosimstudio.github.io/>
- GitHub：<https://github.com/robosimstudio/robosimstudio>
- 技术报告：<https://arxiv.org/abs/2610.18835>
