# robosuite

> 来源归档（Humanoid Motion Intelligence 开源项目主表 + robosuite 官方资料）

- **标题：** robosuite
- **类型：** repo / MuJoCo robot-learning simulator
- **技术路线分组：** 工程与实机部署（上游主表分类）
- **官方仓库：** <https://github.com/ARISE-Initiative/robosuite>
- **项目页：** <https://robosuite.ai/>
- **官方文档：** <https://robosuite.ai/docs/>
- **论文：** [robosuite: A Modular Simulation Framework and Benchmark for Robot Learning](https://arxiv.org/abs/2009.12293)
- **白皮书：** <https://robosuite.ai/assets/whitepaper.pdf>
- **LIBERO 中的依赖声明：** [requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) 固定 `robosuite==1.4.0`
- **LIBERO 安装文档：** <https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html>
- **入库日期：** 2026-07-30；2026-10-03 补充官方文档、论文和 LIBERO 关系
- **一句话说明：** 用 MuJoCo 组合机器人模型、场景、物体与操作任务的模块化仿真框架，提供机器人控制器、传感器与任务 API，可作为 LIBERO 操作基准所用的仿真层。
- **开源状态：** 官方 GitHub 仓库提供源码；版本、许可与运行要求以仓库及发布说明为准。
- **版本提示：** robosuite 当前官方文档为 v1.5；LIBERO 仓库 requirements 固定使用 v1.4.0。复现实验时按 LIBERO 的版本约束安装，勿将当前文档版本视作 LIBERO 的依赖版本。
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [robosuite 工程实体页](../../wiki/entities/robosuite.md)
- **论文详情页：** [robosuite 论文](../../wiki/entities/robosuite.md)

## 与 LIBERO 的关系

LIBERO 的安装文档将 robosuite 描述为其底层仿真环境，当前 requirements 文件明确固定版本 `1.4.0`。robosuite 提供机器人与场景仿真能力；LIBERO 在其上定义语言条件操作任务、演示数据和 lifelong-learning 评估协议。两者是仿真框架与基准的关系，具体兼容性以 LIBERO 当前安装说明和依赖文件为准。

## 为什么值得保留

上游项目主表将 robosuite 列为工程/研究入口；官方文档、项目论文及 LIBERO 对该依赖的固定版本，补足了其仿真定位和基准关联，便于从项目实现继续查阅任务与版本约束。

## 对 wiki 的映射

- [robosuite 工程实体](../../wiki/entities/robosuite.md)
- [robosuite 论文实体](../../wiki/entities/robosuite.md)
- [LIBERO 项目实体](../../wiki/entities/libero-benchmark.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
