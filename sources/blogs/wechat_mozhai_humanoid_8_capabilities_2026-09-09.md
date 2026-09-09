# 一文讲透人形机器人 8 个关键能力：感知、抓取、全身控制、平衡、VLA、世界模型、数据、仿真

> 来源归档（blog / 微信公众号）

- **标题：** 一文讲透人形机器人 8 个关键能力：感知、抓取、全身控制、平衡、VLA、世界模型、数据、仿真
- **类型：** blog
- **作者：** 猫先生M（微信公众号「魔方AI空间」）
- **原始链接：** https://mp.weixin.qq.com/s/-33IGrRqnxM6ALuI5ynODg
- **发表日期：** 未在页面元数据暴露（入库日 2026-09-09）
- **入库日期：** 2026-09-09
- **抓取方式：** WebFetch 直拉 `mp.weixin.qq.com` 正文（Camoufox 工具链本环境未预装）
- **原始抓取落盘：** [`sources/raw/wechat_mozhai_humanoid_8_capabilities_2026-09-09.md`](../raw/wechat_mozhai_humanoid_8_capabilities_2026-09-09.md)
- **系列：** 【从零走向 AGI】（https://ai-mzq.github.io/From-Zero-to-AGI/ ）
- **一句话说明：** 用「红杯入水槽」任务串起人形 **八大能力地图**——感知/抓取/全身控制/平衡为身体底座，VLA/世界模型为智能中枢，数据/仿真为工程飞轮；文内案例**映射到已有 wiki 节点**，不重复造页。

## 核心摘录（归纳，非全文）

### 任务动机

一段 Demo 只展示几秒流畅动作；真实部署难在 **连续、稳定、可恢复** 的物理行动。文内把能力分为三层：

| 层 | 能力 | 文内角色 |
|----|------|----------|
| 基础 | 感知、抓取、全身控制、平衡 | 让身体能接触世界且不摔 |
| 智能中枢 | VLM、VLA、世界模型、任务规划 | 把语言/视觉理解变成可执行意图 |
| 工程底座 | 数据、仿真、后训练、安全约束 | 低成本试错与持续进化 |

### 八能力 → 站内节点（复用，不新建）

| # | 能力 | 文内要点 | 本库节点 |
|---|------|----------|----------|
| 1 | 感知 | RGB-D + 位姿 + 本体 + IMU；主动换视角 | [具身感知六表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md)、[Gemini Robotics](../../wiki/entities/gemini-robotics.md) |
| 2 | 抓取 | 接近→接触→力觉→滑移修正；触觉是第二套眼 | [抓取知识链 hub](../../wiki/overview/hub-grasp.md)、[接触力控 hub](../../wiki/overview/hub-contact-force-control.md) |
| 3 | 全身控制 | 手-躯干-腿协同；MPC/WBC 仍是底座 | [WBC](../../wiki/concepts/whole-body-control.md)、[WBC hub](../../wiki/overview/hub-wbc.md)、[身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) |
| 4 | 平衡 | IMU/足底力矩；负载改变质心 | [Balance recovery](../../wiki/tasks/balance-recovery.md)、[Humanoid locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 5 | VLA | RT-2 token → 连续 flow/chunk；OXE/Octo/OpenVLA | [VLA](../../wiki/methods/vla.md)、[Open X-Embodiment](../../wiki/concepts/open-x-embodiment.md)、[π0](../../wiki/methods/π0-policy.md) |
| 6 | 世界模型 | 动作后果预测；规划/恢复/安全筛查 | [生成式 WM](../../wiki/methods/generative-world-models.md)、[WAM](../../wiki/concepts/world-action-models.md)、[动作后果地图](../../wiki/overview/robot-world-models-action-consequence-technology-map.md) |
| 7 | 数据 | 长尾+失败恢复；质量>数量 | [Teleoperation](../../wiki/tasks/teleoperation.md)、[Action chunking](../../wiki/methods/action-chunking.md)、[Data flywheel](../../wiki/concepts/data-flywheel.md) |
| 8 | 仿真 | Isaac/RoboCasa/Habitat；真机↔仿真闭环 | [Sim2Real](../../wiki/concepts/sim2real.md)、[训练栈分层地图](../../wiki/overview/robot-training-stack-layers-technology-map.md) |

文内系统示例（RT-2、GR00T、Figure Helix、Unitree G1 等）**已有实体或方法页**，本 ingest 只挂接，不新建空壳。

### 文内收束（可执行）

1. **评估看恢复，不看单次 Demo**：换物体/场地/干扰后的成功率、碰撞、延迟与安全距离。
2. **感知错则全栈错**：反光、遮挡、抽屉状态误判会级联到抓取与规划。
3. **VLA 不替代接触控制**：语义泛化与连续力控是互补层（RT-2 vs π0 叙事）。
4. **世界模型放在预测环**：实时观测纠偏 + 硬安全约束兜底，不单独承担毫秒控制。
5. **数据与仿真是飞轮**：真机失败 → 仿真扩展 → 受控验证 → 日志回训（与 [Sim2Real](../../wiki/concepts/sim2real.md) 一致）。

## 对 wiki 的映射

- 写回：[人形八大能力技术地图](../../wiki/overview/humanoid-eight-capabilities-technology-map.md)（**父节点**）
- 交叉：[身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[训练栈分层](../../wiki/overview/robot-training-stack-layers-technology-map.md)、[VLA+WM 阅读路线](../../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md)
- **本次未**为文内厂商 Demo（Optimus、远征 A2 等）新建实体：已有页只回链。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 八能力对照既有 wiki（复用 / 待升格，**0 重复造页**）
- [x] 升格 overview 技术地图
