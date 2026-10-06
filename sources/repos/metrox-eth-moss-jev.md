# MOSS × Jev 仿真（metrox-eth/moss-jev）

> 来源归档

- **标题：** MOSS × Jev — recorded physics replay
- **类型：** repo / 仿真与演示
- **来源：** Show Robotics / metrox-eth；决策模型 Jev by TypeSafe AI
- **链接：** <https://github.com/metrox-eth/moss-jev>
- **入库日期：** 2026-10-06
- **核查日期：** 2026-10-06
- **一句话说明：** 提供 MOSS 的 MuJoCo 简化模型、可在 CPU 上运行的仿真任务，以及浏览器端的录制物理回放。
- **项目页：** [Show Robotics · MOSS](../sites/showrobotics-moss.md)
- **主仓库：** [metrox-eth/moss](./metrox-eth-moss.md)
- **沉淀到 wiki：** [MOSS 实体页](../../wiki/entities/moss.md)

## 仓库内容与边界

- **浏览器演示：** V0.2 的三个任务曾在 MuJoCo 中运行，并用 Jev API 生成动作决策；浏览器按 25 fps 回放已录制轨迹。页面不进行实时推理、API 请求或后端计算。
- **本地复现：** live/ 提供 CPU 仿真器、便携 MOSS MJCF / 网格与三个已录制任务；回放不需要 API key 或 GPU。
- **模型精度：** V0.2 使用简化底盘平移与夹爪物理，不是制造 CAD，也不是经标定的履带驱动模型。静态浏览器演示仍为录制回放。
- **许可与资产：** 仓库 README 指向 licenses/NOTICE.txt，其中区分 SO-101 与夹爪几何、Three.js 等第三方组件许可；具体再分发应遵循对应 NOTICE。

## 推荐入口

- 仓库说明：[README.md](https://github.com/metrox-eth/moss-jev/blob/main/README.md)
- 本地仿真：[live/README.md](https://github.com/metrox-eth/moss-jev/blob/main/live/README.md)
- 模型集成契约：[live/INTEGRATION.md](https://github.com/metrox-eth/moss-jev/blob/main/live/INTEGRATION.md)
- 浏览器演示：<https://www.showrobotics.ai/moss-jev/>

## 对 wiki 的映射

- 项目页：[showrobotics-moss.md](../sites/showrobotics-moss.md)
- MOSS 主仓：[metrox-eth-moss.md](./metrox-eth-moss.md)
- 实体页：[moss.md](../../wiki/entities/moss.md)
