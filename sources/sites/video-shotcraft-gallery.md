# video-shotcraft Gallery（项目页）

> 来源归档（site / project page）

- **标题：** video-shotcraft Gallery — live motion previews
- **类型：** project page / GitHub Pages
- **URL：** <https://vincentwei1021.github.io/video-shotcraft/>
- **代码：** <https://github.com/Vincentwei1021/video-shotcraft>
- **作者：** Vincentwei1021
- **核查日期：** 2026-11-12（官方 Gallery 页面 124 Cards / 150 Styles / 150 Samples）
- **一句话说明：** video-shotcraft 的在线样片画廊：搜索、筛选、切换样式，并复制镜头配方卡名称，供 Agent Skill 制作前选型。

## 开源状态（2026-11-12 复核；部署信息沿用历史归档）

- Gallery HTTP 200；站点源在主仓 `gallery/`，由 `.github/workflows/deploy-pages.yml` 部署。
- 页面提供 **library** 浏览、`llms.txt`、多语言（EN/CN/JA README 互指）。
- **代码入口**指向官方仓 [Vincentwei1021/video-shotcraft](https://github.com/Vincentwei1021/video-shotcraft)（Apache-2.0）。
- **结论：已开源**（技能库 + Gallery + Ink Press 模板）。详见 [`sources/repos/video-shotcraft.md`](../repos/video-shotcraft.md)。

## 核心摘录（归纳）

- **用途：** 在写 Remotion 代码前浏览 **150** 条动态样片 / **150** 个样式（**124** 张配方卡），选定卡名与 `style-key`。
- **与技能联动：** `SKILL.md` 要求用卡时先经 `gallery/api/library.json` 校验，再读配方卡与准确 demo TSX——Gallery 是选型 UI，不是替代本地源码。
- **媒体托管（历史）**：2026-07-26 起 preview mp4 移出 git 改存 release；本次复核以当前 Gallery 所示数量为准，不据此推断目前所有媒体 URL 的托管方式。

## 版本变更

- **2026-07-31 历史值：** 104 卡、161 样式/预览；与当前值不同。
- **2026-10 官方更新：** [README](https://github.com/Vincentwei1021/video-shotcraft/blob/main/README.md) 宣布镜头刷新，当前页面直接显示 124 Cards / 150 Styles / 150 Samples。
- **操作提示：** 先在 Gallery 选择样式并复制镜头名，再以 `gallery/api/library.json` 校验卡名与 `style-key`，阅读对应配方卡与 demo TSX 后再制作。

## 对 wiki 的映射

- [video-shotcraft 实体页](../../wiki/entities/video-shotcraft.md)
- [仓库源归档](../repos/video-shotcraft.md)

## 参考来源（原始）

- Gallery：<https://vincentwei1021.github.io/video-shotcraft/>
- 代码：<https://github.com/Vincentwei1021/video-shotcraft>
