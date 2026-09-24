# AtlasOS 项目站与文档

> 来源归档

- **标题：** AtlasOS — Optimize Windows for performance, privacy and usability
- **类型：** site
- **链接：** <https://atlasos.net>
- **文档：** <https://docs.atlasos.net>
- **Discord：** <https://discord.atlasos.net>
- **代码：** <https://github.com/Atlas-OS/Atlas>
- **入库日期：** 2026-09-24
- **一句话说明：** Atlas 官方站点与文档：安装 Playbook、安全选项说明、贡献与测试指南；不重分发 Windows 镜像，合规 Microsoft 使用条款。
- **沉淀到 wiki：** [`wiki/entities/atlas-os.md`](../../wiki/entities/atlas-os.md)

## 开源状态（项目页核查，2026-09-24）

**已开源**：GitHub 主仓 GPL-3.0 Playbook；文档站公开安装/FAQ/贡献流程。安装需自备合法 Windows 介质，Atlas 仅提供可审计的 Playbook 修改包。

## 产品要点（文档摘录）

| 维度 | 说明 |
|------|------|
| **交付形态** | AME Wizard Playbook（`.apbx`，ZIP 密码 `malte`），非定制 ISO |
| **隐私** | 移除多数 Windows 遥测，组策略减采集；浏览器等第三方不在范围 |
| **性能** | debloat、后台应用、MMCSS、分页等可配置 tweak；强调稳定优于 placebo |
| **安全** | Defender/SmartScreen/Update/UAC/核心隔离等 **可选**，文档列 pros/cons |
| **合规** | 不修改 Windows 激活；不再分发修改版系统镜像 |

## 安装与文档入口

- [Installation](https://docs.atlasos.net/getting-started/installation/)
- [Install FAQ / Removed features](https://docs.atlasos.net/install-faq/removed-features/)
- [Atlas and security](https://docs.atlasos.net/general-faq/atlas-and-security/)
- [Building the Playbook](https://docs.atlasos.net/contributing/playbook/)

## 交叉链接

- [Atlas 仓库归档](../repos/atlas_os_atlas.md)
- [wiki/entities/atlas-os.md](../../wiki/entities/atlas-os.md)
