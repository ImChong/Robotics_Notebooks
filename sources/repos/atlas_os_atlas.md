# Atlas（Atlas-OS/Atlas 代码仓库）

> 来源归档

- **标题：** Atlas — open and lightweight modification to Windows
- **类型：** repo
- **组织：** Atlas-OS
- **链接：** <https://github.com/Atlas-OS/Atlas>
- **官网：** <https://atlasos.net>
- **文档：** <https://docs.atlasos.net>
- **入库日期：** 2026-09-24
- **许可：** GPL-3.0
- **Stars：** ~21k（2026-09）
- **一句话说明：** 开源 Windows 优化 Playbook：经 AME Wizard 应用隐私/性能/可用性 tweak，不重打包 Windows ISO；CI 在 `src/sxsc` 变更时自动构建 CAB 并 bot 回推 PR。
- **沉淀到 wiki：** [`wiki/entities/atlas-os.md`](../../wiki/entities/atlas-os.md)

## 开源状态

**已开源**：Playbook 配置（`src/playbook/Configuration/`）、sxsc YAML、构建脚本与文档均在 GitHub；AME Wizard GUI 闭源，后端 [TrustedUninstaller CLI](https://github.com/Ameliorated-LLC/trusted-uninstaller-cli)（MIT）与 [utilities](https://github.com/Atlas-OS/utilities) 二进制开源可查 hash。

## 仓库结构（维护入口）

| 路径 | 说明 |
|------|------|
| `src/playbook/Configuration/` | Playbook 主配置：debloat、performance、networking、security 等 YAML |
| `src/sxsc/` | sxsc 包定义 YAML；变更触发 CI 构建 `.cab` |
| `src/playbook/Executables/AtlasModules/` | 内置模块与 CI 产出的 CAB |
| `.github/workflows/apbx.yaml` | 推送 `src/**` 时构建 Playbook（`.apbx`）artifact；sxsc 变更时 **自动 commit + push CAB** |
| `.github/workflows/labeler.yaml` | PR 自动打 label |

## CI：自动回推 CAB（PR 内 bot 合并）

`apbx.yaml` 在检测到 `src/sxsc/*.yaml` 变更（或 `regenAllConfigs` 标记）时：

1. clone [Atlas-OS/sxsc](https://github.com/Atlas-OS/sxsc)，按 YAML 生成并签名 CAB；
2. 将 CAB 复制到 `AtlasModules/Packages/`；
3. 以 `atlasos-admin` bot **commit 并 push** 回当前分支（commit message：`feat: auto-update CAB packages`）；
4. 打包 password=`malte` 的 `.apbx` Playbook 为 workflow artifact。

贡献者只需改 sxsc 文本配置，二进制 CAB 由 CI 自动合并进 PR，无需本地 Windows 签名环境。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [atlas-os](../../wiki/entities/atlas-os.md) | 实体页 |
| [atlasos 项目站](../sites/atlasos.md) | 官网与文档摘录 |
| [winui](../../wiki/entities/winui.md) | Windows 工控机 HMI 栈（Atlas 优化后的宿主 OS） |
| [teleoperation](../../wiki/tasks/teleoperation.md) | 遥操作 Windows 工控机场景 |

## 参考

- [Atlas README](https://github.com/Atlas-OS/Atlas/blob/main/README.md)
- [Installation](https://docs.atlasos.net/getting-started/installation/)
- [Contribution Guidelines](https://docs.atlasos.net/contributing/contribution-guidelines/)
