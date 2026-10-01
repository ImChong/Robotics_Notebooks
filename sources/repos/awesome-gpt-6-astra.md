# Awesome GPT-6 Astra

> 来源归档；核查日期：2026-10-01。

- **类型：** repo / 社区作品目录
- **仓库：** <https://github.com/MartinDelophy/awesome-gpt-6-astra>
- **项目页：** <https://astragames.aigccreative.com/en>；见[站点归档](../sites/astra-games.md)
- **代码：** 仓库中的 `website/`、`works/orbital-garden/`、`works/thunderfall/` 等；目录中另有第三方源码链接。
- **论文 / 权重 / 数据集：** 本次来源未提供机器人论文、模型权重或机器人训练数据入口；目录 JSON 是展示数据。
- **入库日期：** 2026-10-01
- **一句话说明：** 社区维护的 AI 辅助浏览器游戏与交互作品清单，附作者、体验入口、模型参与记录及可用开发资源。
- **为什么保留：** 用于机器人知识站交互教学和浏览器展示的设计参考，而非机器人控制算法依据。
- **Wiki：** [Astra 交互作品集](../../wiki/entities/astra-games.md)

## 来源核查与开放边界

README 于本次核查时标注目录更新日为 2026-09-30、165 个作品；数量是时间快照。README 明确说明收录不要求开源，模型归因依赖作者或提交者声明，社区与 OpenAI 无隶属关系，也不构成模型基准。

| 范围 | 核查结论 | 入口 |
| --- | --- | --- |
| 清单与原创编排 | 公开，仓库说明原创文字与美术采用 CC0 1.0 | [README](https://github.com/MartinDelophy/awesome-gpt-6-astra/blob/main/README.md)、[LICENSE](https://github.com/MartinDelophy/awesome-gpt-6-astra/blob/main/LICENSE) |
| 展示站实现 | 已公开，README 给出开发、测试和构建入口 | [website/README.md](https://github.com/MartinDelophy/awesome-gpt-6-astra/blob/main/website/README.md) |
| Orbital Garden | 已公开，单文件 HTML、Prompt、元数据；目录原创内容声明 CC0 | [作品 README](https://github.com/MartinDelophy/awesome-gpt-6-astra/blob/main/works/orbital-garden/README.md) |
| THUNDERFALL | 已公开，ES Modules、引擎测试、Prompt、设计文档；原创内容声明 CC0 | [作品 README](https://github.com/MartinDelophy/awesome-gpt-6-astra/blob/main/works/thunderfall/README.md) |
| 全部第三方作品 | 仅部分条目列源码；按作品独立核查许可 | [上游目录](https://github.com/MartinDelophy/awesome-gpt-6-astra#games) |

目录的 CC0 不覆盖所有外链游戏及第三方素材；例如 README 对 Toy2Game 单独说明非商业使用许可。

## 核心摘录

1. **目录结构可复用：** 每条记录包含作品、作者、访问要求、实际截图、模型参与证据及开发资源；便于区分可体验、可复现、可再利用。
2. **展示站与目录解耦：** `website/` 解析上游 README，经 API 缓存供前端读取；失败保留最近成功数据或 fallback 快照，标注旧数据时间。
3. **Orbital Garden 是图形交互案例：** 原生 WebGL 粒子、形态连续切换、暂停、调速、参数化海报导出；README 明确其引力为艺术想象，不是天体力学或流体模拟。
4. **THUNDERFALL 是工程验证案例：** Canvas 2D + Web Audio；HTTP 静态服务运行，Node 内置模块测试；README 区分引擎模拟、浏览器检查和实物手机验证。
5. **模型参与不等于一次生成：** 两个作品都记录了迭代与协作审阅，不能据此推断模型在机器人软件上的性能。

## 对 wiki 的映射

- [Astra 交互作品集](../../wiki/entities/astra-games.md)：资源定位、交互复用与验证边界。
- [Easy-Vibe](../../wiki/entities/easy-vibe.md)：系统教程与成品案例互补。
- [Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)：从可演示到可维护的判断框架。

本次核查公开目录、源码路径及 README 运行说明，未逐一运行全部游戏，也未独立复测作者记录的测试结果。
