# LejuBFM 官方资料：通用小脑运控基座

> 来源归档（乐聚官方主页与技术文档）

- **项目主页：** <https://www.lejubfm.lejurobot.com/>
- **整体架构：** <https://www.lejubfm.lejurobot.com/docs/overview/architecture>
- **基模环境配置：** <https://www.lejubfm.lejurobot.com/docs/bfm/setup>
- **离线轨迹追踪部署：** <https://www.lejubfm.lejurobot.com/docs/bfm/deployment>
- **OpenLET 小脑动作集：** <https://www.lejubfm.lejurobot.com/docs/datasets/openlet>
- **数据申请入口：** <https://openlet.openatom.tech/data/let-humanmotion-roban2>
- **代码仓库：** <https://atomgit.com/OpenLET/Lejulab_ARM>
- **入库日期：** 2026-10-17
- **一句话说明：** 乐聚官方通用小脑运控基座模型与工具链资料，涵盖 sonic/heft 模型、Roban 2.2 部署、全身遥操与 OpenLET 动作数据。

## 基模与部署要点

- 官方将 BFM 定位为全身动作跟踪与协调控制基座，围绕通用运控基模、全身遥操、整机数采训练和动作数据集组织开发流程。
- 离线动作追踪文档列出 **sonic** 与 **heft** 两套模型。两者共用环境，但应按版本分别匹配模型、配置与轨迹；文档称其位于 Lejulab_ARM 仓库的 `beta` 分支。
- 官方部署示例限定 **Roban 2.2**（`ROBOT_VERSION=17`）；真机目标是 **RK3588 / ARM64**，建议 GCC/G++ 11；仿真使用开发机 Docker 与 MuJoCo。
- 官方流程为选模型 → 配置配套轨迹 → MuJoCo 仿真验证 → 真机播放。模型路径、观测参数、关节顺序、轨迹格式不可跨版本猜测或混配；高动态动作需按文档配置保护措施。
- 当前公开资料主要说明离线轨迹追踪与部署。未见该站点披露模型架构、训练配方、定量评测协议或跨本体零样本迁移证据。

## OpenLET 小脑动作集

- 乐聚主页称动作集覆盖 **800+ 动作类别、39,692 条动作数据**；官方数据页称其包括非交互、环境交互、物体交互、武术与舞蹈五类，动作重定向至鲁班机型并按 **120 fps** 重采样。
- 数据采用申请制；申请资格、获取方式和使用条款以 OpenLET 平台及审批结果为准。代码仓库可见不等于数据授权，也不代表模型权重和数据均可无条件下载。
- 代码与数据入口分开：Lejulab_ARM 提供运行平台和模型/配置资料入口；LET-HumanMotion-Roban2 数据集另行申请。

## 对 wiki 的映射

- [LejuBFM 运控基座实体](../../wiki/entities/lejubfm-bfm.md)
- [OpenLET LET-HumanMotion-Roban2](../../wiki/entities/openlet-humanmotion-roban2.md)
- [OpenLET 社区](../../wiki/entities/openlet.md)
- [Lejulab_ARM 仓库归档](../repos/lejulab-arm.md)
