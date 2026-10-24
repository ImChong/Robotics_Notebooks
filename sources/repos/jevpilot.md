# JevPilot（standardagents/jevpilot）

> 来源归档

- **标题：** A playable Three.js driving simulator with Jev-powered autopilot
- **类型：** repo / 浏览器驾驶仿真
- **来源：** Standard Agents
- **链接：** <https://github.com/standardagents/jevpilot>
- **许可：** GitHub 仓库未声明许可证
- **项目页：** <https://jevpilot.standardagents.ai/>
- **入库日期：** 2026-10-24
- **核查日期：** 2026-10-24
- **一句话说明：** Three.js 可玩驾驶模拟器，由 Jev 在预先采样的转向与速度组合中选择行驶动作。
- **关联 wiki：** [Jev（TypeSafe）](../../wiki/entities/typesafe-jev.md)

## 仓库内容与边界

- **决策输入：** 候选路径、道路边界、附近车辆、信号灯、停车状态记忆与目的地指引以紧凑表格发送给 Jev；详细几何与控制计算留在本地代码。
- **调用频率：** README 描述转弯或交通繁忙时最高约 4 次/秒，路况清晰时约 1.5 次/秒；只有一个可选答案的问题在本地解决。
- **运行方式：** 本地开发可用自己的 TypeSafe API key 启动，密钥由服务端读取；托管试玩使用登录会话与免费额度，API key 不发给浏览器。
- **复现边界：** 这是 Three.js 驾驶模拟器，不是真车道路测试。仓库未声明开源许可证，使用与再分发前应自行核对权利。

## 推荐入口

- 仓库说明：[README.md](https://github.com/standardagents/jevpilot/blob/main/README.md)
- 在线模拟器：<https://jevpilot.standardagents.ai/>
- [项目演示站归档](../sites/jevpilot.md)
