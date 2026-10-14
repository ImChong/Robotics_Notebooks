# Physical Intelligence Pi API beta

> 来源归档（官方公告与 API beta 申请页）

- **标题：** Frontier Models for Every Robot: Announcing our API beta
- **类型：** official blog / hosted robotics model API
- **公告：** <https://www.pi.website/blog/api-beta>
- **申请：** <https://www.pi.website/waitlist>
- **关联开源模型仓库：** <https://github.com/Physical-Intelligence/openpi>
- **组织：** Physical Intelligence
- **发布日期：** 2026-10-13
- **入库日期：** 2026-10-14
- **服务源码：** 官方公告和申请页未链接 API 服务端源码；`openpi` 是相关模型的开源代码仓，不是托管 API 实现。
- **一句话说明：** Pi API beta 提供托管机器人模型微调与推理：用户上传机器人数据，在 Physical Intelligence 管理的算力上微调，再经托管 gateway 为机器人提供推理。

## 官方公告要点

- API 的目标是降低适配机器人基础模型所需的 ML 基础设施、算力与数据管线工作量。
- 公开流程为上传数据、启动托管微调、通过 hosted gateway 运行微调后的策略，并用评测数据决定下一轮采集与训练。
- 早期合作覆盖重型设备、农业、餐饮、制造与服务机器人等应用；官方称多数展示案例采用云端托管推理，并提到部分部署遇到网络连接挑战。
- 官方以 Almond 的 Axol 双臂机器人为例，称团队通过 Pi API 迭代四天后获得了用于去毛刺任务的模型。
- API 仍处于 beta，官方正在分批接入；申请表会询问机器人类型、采集数据、目标任务、团队经验、部署位置与成功标准。

## 公开状态边界（截至 2026-10-14）

- **托管 API：** beta 申请制，官方尚未公开可直接调用的 API reference、endpoint / schema、SDK、价格或 SLA。
- **模型代码：** 官方 [openpi](https://github.com/Physical-Intelligence/openpi) 仓库提供 π₀、π₀-FAST、π₀.₅ 的代码和示例，可自行部署；它与新发布的托管 Pi API 是不同交付方式。
- **服务实现：** 公告与申请页面没有链接 API 服务端实现，因此服务本身目前按未公开处理。

## 关联归档

- 官方模型代码：[openpi](../repos/openpi.md)
- π₀ 项目页：[π₀](https://www.pi.website/blog/pi0)
- π₀.₅ 项目页：[π₀.₅](https://www.pi.website/blog/pi05)
- Wiki 实体：[Pi API beta](../../wiki/entities/physical-intelligence-pi-api.md)

