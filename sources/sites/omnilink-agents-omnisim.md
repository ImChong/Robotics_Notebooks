# omnilink-agents.com/omnisim（OmniSim 产品页）

> 来源归档（site · OmniLink 仿真器营销与 clone 入口）

- **标题：** OmniSim — Talk to a simulator
- **类型：** site / product-portal
- **URL：** <https://www.omnilink-agents.com/omnisim>
- **机构：** OmniLink
- **配套代码：** <https://github.com/omnilink-tech/omnisim> — [`sources/repos/omnisim.md`](../repos/omnisim.md)
- **入库日期：** 2026-08-28
- **一句话说明：** OmniSim 的产品页：把仿真器定位成「对编码代理说话」的 OmniLink 界面；引导 `git clone --recurse-submodules`、仓库 `AGENTS.md` 与 `python -m omnisim doctor`。

## 开源状态（步骤 2.5，截至 2026-08-28）

页内 CTA：**Clone on GitHub** / **Apache 2.0. Free, forever.** 明确链到 [omnilink-tech/omnisim](https://github.com/omnilink-tech/omnisim)，并声明独立 fork of Webots、保留上游版权、名称与 orb 为商标。

**结论：已开源（Apache-2.0）。** 无独立数据集或权重门户；复现入口是 GitHub 仓库而非本页附件。

## 页面结构要点（2026-08-28 抓取）

- **定位：** *the OmniLink interface*。clone 后在编码代理中打开，由代理自行安装、建世界、加机器人并接到 agent。
- **推荐首句：** “Set up OmniSim from this fresh clone — install whatever the toolchain needs, build it, and launch the warehouse Husky demo.”
- **运行叙事：** 先 `python -m omnisim doctor`，再 `python -m omnisim run-world projects/samples/demos/worlds/chat/omnilink_husky.wbt`；右键机器人 → Show Robot Window → 输入 “forward 1 meter”。离线命令路由处理固定短语，无需账号。
- **硬件叙事（本页）：** clone 即安装，本机编 C++/Qt6；Newton GPU 需 NVIDIA/CUDA，**无卡则走 ODE CPU 回退**。
- **资产叙事（本页）：** 「18 robots from 11 makers」（Husky/Jackal、Rosbot、TurtleBot3、Spot、Go2/B2/G1/H1、Valkyrie、UR3e–UR10e、Mavic、Digit、Panda、Neura LARA5/Maira）；原生 URDF；14 个 chat worlds；`omniworld` 按 seed 生成 Mars/森林/仓库/室内。
- **许可：** Apache-2.0；独立 fork，非 Cyberbotics 附属；改 fork 须改名。
- **后续：** 设 `OMNI_KEY` 后机器人作为 agent profile 注册到 OmniLink 平台。

## 与仓库 README 不对齐之处

本页仍宣传 **ODE CPU 回退** 与更宽的机型名单（Spot / Digit / Valkyrie / Franka 等），世界路径示例仍用 **`.wbt`**。仓库 README / `AGENTS.md`（入库日）写 **Newton 唯一后端、ODE 已删、世界以 `.omniworld` 为准**，机器人表也更窄。

选型与复现 **以 [`sources/repos/omnisim.md`](../repos/omnisim.md) 为准**；本页只保留产品定位、clone 话术与开源声明。

## 为何值得保留

- 证明官方项目页存在且 **代码链接可核**（步骤 2.5 不以 PDF/口头承诺为准）。
- 记录营销页与引擎事实源的漂移，避免 wiki 把已删除的 ODE 回退写成当前能力。

## 对 wiki 的映射

- [`wiki/entities/omnisim.md`](../../wiki/entities/omnisim.md)
- [`sources/repos/omnisim.md`](../repos/omnisim.md)
