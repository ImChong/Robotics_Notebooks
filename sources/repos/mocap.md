# MoCaP（MoCAP-7/MoCAP）

> 来源归档（repo）

- **标题：** MoCAP
- **类型：** repo
- **链接：** <https://github.com/MoCAP-7/MoCAP>
- **项目页：** <https://mocap-7.github.io/MoCAP-page/>
- **论文：** <https://arxiv.org/abs/2610.20998>
- **默认分支：** `master`
- **许可：** 未注明（截至 2026-10-23，仓库根目录无 `LICENSE` 文件）
- **入库日期：** 2026-10-23
- **一句话说明：** MoCaP 论文的公开实现，包含被动视频空间记忆规划器、代码代理和 YOR 移动操作服务接口，以及 CoW、ApexNav 等导航基线。
- **沉淀到 wiki：** [`wiki/entities/paper-mocap-mobile-code-as-policies.md`](../../wiki/entities/paper-mocap-mobile-code-as-policies.md)

## 仓库入口

| 路径 / 命令 | 作用 |
|------|------|
| `nav_planner/` | 被动视频空间记忆与互动事件处理；README 提供 `nav-planner`、`nav-planner-readiness` 命令 |
| `yor_agent/` | Code-as-Policy 代理、机器人原语与服务；入口命令为 `yor-agent` |
| `baselines/mobilemanip/capx/` | CaP-X 移动操作基线 |
| `baselines/nav/cow/`、`baselines/nav/apexnav/` | CoW 与 ApexNav 物体导航基线 |
| 安装 | `pip install -e ./nav_planner -e './yor_agent[manipulation]'` |

## 开源边界

仓库公开了视频处理、代码代理与机器人任务接口；README 标注该仓库用于双盲评审。仓库目前没有 LICENSE 文件，许可边界未明确。运行完整真机任务还需要 API 凭据、机器人服务和硬件配置。

## 与知识库节点的关系

- [MoCaP 论文实体页](../../wiki/entities/paper-mocap-mobile-code-as-policies.md)
- [项目页归档](../sites/mocap-project.md)
