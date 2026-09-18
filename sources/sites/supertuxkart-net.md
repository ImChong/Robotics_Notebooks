# SuperTuxKart（项目主页）

> 来源归档（ingest 关联资料）

- **标题：** SuperTuxKart
- **类型：** site / project-page / game
- **项目主页：** <https://supertuxkart.net/>
- **代码：** <https://github.com/supertuxkart/stk-code> — 见 [`sources/repos/supertuxkart-stk-code.md`](../repos/supertuxkart-stk-code.md)
- **Releases：** <https://github.com/supertuxkart/stk-code/releases>
- **源码编译：** [`INSTALL.md`](https://github.com/supertuxkart/stk-code/blob/master/INSTALL.md)
- **版本控制说明：** <https://supertuxkart.net/Source_control>
- **资产 SVN：** <https://svn.code.sf.net/p/supertuxkart/code/stk-assets>
- **入库日期：** 2026-09-18
- **一句话说明：** 免费开源卡丁车竞速游戏官方站：下载（Linux/Windows/macOS/Android）、社区论坛、博客与 **Git+SVN 双仓** 源码说明。

## 开源核查（步骤 2.5，2026-09-18）

项目主页 Footer / Source control 页明确链到 **GitHub stk-code** 与 **SourceForge SVN stk-assets**。

| 链接 | 用途 |
|------|------|
| `/Download` | 预编译二进制（多平台） |
| GitHub `supertuxkart/stk-code` | **C++ 引擎与游戏逻辑** |
| SVN `stk-assets` | **运行时资产**（赛道、模型、音效等） |
| SVN `media/trunk`（可选） | 艺术家源文件 ~3.2GB，**非游玩必需** |
| [Releases](https://github.com/supertuxkart/stk-code/releases) | 稳定版（截至入库日最新 **1.5**，2025-10-20） |

**结论：** **已开源**（GPL，见仓内 `COPYING`）；游玩/编译需 **code + assets** 两仓并排目录。

## 关联 wiki

- [`wiki/entities/supertuxkart.md`](../../wiki/entities/supertuxkart.md)
