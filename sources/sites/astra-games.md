# Astra Games 展示站

> 来源归档；核查日期：2026-10-01。

- **类型：** site / 交互作品展示
- **项目页：** <https://astragames.aigccreative.com/en>
- **上游：** <https://github.com/MartinDelophy/awesome-gpt-6-astra>
- **代码：** <https://github.com/MartinDelophy/awesome-gpt-6-astra/tree/main/website>
- **仓库归档：** [Awesome GPT-6 Astra](../repos/awesome-gpt-6-astra.md)
- **入库日期：** 2026-10-01
- **一句话说明：** 从上游 README 获取作品的分类、搜索、体验与创作说明入口。
- **Wiki：** [Astra 交互作品集](../../wiki/entities/astra-games.md)

## 页面与源码核查

英文首页提供 Catalogue、Guides、About & editorial policy、Contact & corrections。首页说明目录来自上游 README，不另存第二份人工清单；本次页面显示 165 项，数字会随目录变化。

**开放结论：** 展示站源码已公开；被收录作品的代码开放程度不统一。源代码入口及运行说明见 [website/README.md](https://github.com/MartinDelophy/awesome-gpt-6-astra/blob/main/website/README.md)，不能把本站可访问解释为每个游戏都可开源复现。

## 可保留的展示机制

- **单一内容来源：** API 解析上游 README；作品新增不要求改 UI 或重新发布页面。
- **更新与故障回退：** website README 描述可见页面每 5 分钟检查，聚焦时按间隔再检查；服务端请求驱动刷新与缓存。失败保留旧目录或快照，并展示状态。不是即时推送。
- **预览降级：** 作者图片、本地截图、页面社交预览、仓库预览卡片、文字占位依次回退；作品缺图仍保留入口。
- **作者归因与校正：** 体验链接、作者来源和模型参与记录分开；提供纠错入口。

## 对机器人知识站的启发（本库归纳）

可以沿用“稳定内容来源 + 展示层 + 明示旧数据”的组织方法，但 Robotics_Notebooks 已有 wiki/export 链路，应复用现有导出结构，避免新建并行人工目录。浏览器作品可启发参数交互与可视化设计；物理真实性仍须由机器人仿真和评测验证。

本次打开首页并核对仓库文档；未遍历所有指南、测试每个体验入口，也未验证线上运行代码与仓库 HEAD 完全一致。
