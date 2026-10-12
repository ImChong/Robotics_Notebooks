# OpenLET/Lejulab_ARM（LejuBFM 运控代码）

> 来源归档（官方 AtomGit 仓库与 LejuBFM 文档）

- **仓库：** <https://atomgit.com/OpenLET/Lejulab_ARM>
- **代码类型：** C++
- **关联官方文档：** <https://www.lejubfm.lejurobot.com/docs/bfm/setup>
- **关联数据：** [LET-HumanMotion-Roban2](https://atomgit.com/OpenLET/LET-HumanMotion-Roban2.git)（数据需单独申请）
- **一句话说明：** 乐聚公开的 ARM 运控代码仓；LejuBFM 文档称 sonic 与 heft 模型均已上传，提供 Roban 2.2 编译环境和部署步骤。

## 官方公开状态与边界

- 仓库公开可见，AtomGit 仓库页面标注 C++、master 分支；页面未显示明确许可证，商业/再分发许可需以仓库 LICENSE 与权利人说明核验。
- LejuBFM setup 文档要求选择仓库中的配套版本，并针对 Roban 2.2（`ROBOT_VERSION=17`）编译；真机环境为 RK3588 / ARM64 与 gcc/g++ 11。
- 仓库开放不代表配套动作数据开放下载；数据通过 OpenLET 申请入口获取。

## 对 wiki 的映射

- [OpenLET LET-HumanMotion-Roban2](../../wiki/entities/openlet-humanmotion-roban2.md) — 数据集、权限边界及策略使用场景
- [LejuLab-Deploy](../../wiki/entities/cn-os-lejulab-deploy.md) — 乐聚部署控制相关仓
