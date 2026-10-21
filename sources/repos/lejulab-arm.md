# OpenLET/Lejulab_ARM（LejuBFM 运控代码）

> 来源归档（官方 AtomGit 仓库与 LejuBFM 文档）

- **仓库：** <https://atomgit.com/OpenLET/Lejulab_ARM>
- **代码类型：** C++
- **关联官方文档：** <https://www.lejubfm.lejurobot.com/docs/bfm/setup>
- **模型部署文档：** <https://www.lejubfm.lejurobot.com/docs/bfm/deployment>
- **关联数据：** [LET-HumanMotion-Roban2](https://atomgit.com/OpenLET/LET-HumanMotion-Roban2.git)（数据需单独申请）
- **一句话说明：** 乐聚公开的人形机器人运控部署平台；LejuBFM 文档指向该仓库中的 sonic/heft 基模、配套轨迹与 Roban 2.2 部署配置。

## 官方公开状态与边界

- AtomGit 仓库公开可访问；仓库页面默认显示 `master`，而 LejuBFM 部署文档注明 **sonic/heft 模型资源位于 `beta` 分支**。检查模型、配置和 CSV 轨迹时应切到文档要求的分支，并使用彼此匹配的版本。
- 仓库提供乐聚机器人仿真与实物控制支持；LejuBFM 当前部署步骤针对 Roban 2.2（`ROBOT_VERSION=17`），真机目标为 RK3588 / ARM64，需 gcc/g++ 11。
- 配套程序可用于 MuJoCo 仿真和真机轨迹播放；文档要求模型与配置、关节顺序、输入参数及轨迹格式严格配对。
- 仓库页面未显示明确许可证；商业使用与再分发需依据仓库实际 LICENSE 和权利人说明确认。
- 动作数据不随代码仓库自动开放下载；OpenLET 数据需单独申请，按平台条款及审批结果使用。

## 对 wiki 的映射

- [LejuBFM 通用小脑运控基座](../../wiki/entities/lejubfm-bfm.md) — 模型能力、部署边界与复现路径
- [OpenLET LET-HumanMotion-Roban2](../../wiki/entities/openlet-humanmotion-roban2.md) — 数据集、权限边界及策略使用场景
- [LejuLab-Deploy](../../wiki/entities/cn-os-lejulab-deploy.md) — 乐聚部署控制相关仓
