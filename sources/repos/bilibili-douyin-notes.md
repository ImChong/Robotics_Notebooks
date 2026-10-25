# bilibili-douyin-notes

> 来源归档

- **标题：** bilibili-douyin-notes
- **类型：** repo（视频转课堂笔记 Agent Skill）
- **作者：** lucky-Luor
- **仓库：** <https://github.com/lucky-Luor/bilibili-douyin-notes>
- **许可证：** MIT
- **平台：** Bilibili、抖音、YouTube、小红书
- **入库日期：** 2026-10-25
- **沉淀到 wiki：** 是 → [bilibili-douyin-notes](../../wiki/entities/bilibili-douyin-notes.md)

---

## 项目定位

遵循 Agent Skills 规范的开源技能：给代理一个视频链接，按平台抓取元数据与字幕；缺少字幕时使用本机 `faster-whisper` 语音识别，再结合视频截图和课堂笔记模板生成 Markdown。仓库 README 给出的课堂笔记样例是“局部最小值与鞍点”，与截图中的课程笔记示例一致。

## 主要能力

| 阶段 | 实现 |
|------|------|
| 视频入口 | 识别 B 站、抖音、YouTube、小红书链接与分享文本；YouTube 优先获取字幕 |
| 转写 | 字幕优先；没有字幕时用 `faster-whisper` 本地 ASR，CPU int8 可用，不调用付费 ASR API |
| 画面 | 可按转写时间戳用 PyAV 抽取重点截图，并做黑屏、空屏和模糊帧初筛 |
| 笔记 | 输出 Markdown，包含分节要点、术语、代码引用、知识地图、概念卡、自测题与视频时间索引 |
| 安装 | 兼容 Agent Skills 的代理；README 列出 Claude Code、ZCode、DSH、Codex 等技能目录 |

B 站登录是可选配置，主要用于获取字幕；抖音、小红书的 README 流程称无需账号。不同平台仍受视频可访问性、平台策略与本地依赖影响。

## 与本库关系

- [wiki/entities/bilibili-douyin-notes.md](../../wiki/entities/bilibili-douyin-notes.md)：能力边界与使用判断。
- [Agent Skills（Addy Osmani）](../../wiki/entities/agent-skills-addyosmani.md)：通用编码代理技能包，可与面向视频学习的本项目对照。
- README 示例：[局部最小值与鞍点课堂笔记](https://github.com/lucky-Luor/bilibili-douyin-notes/blob/main/examples/%E8%AF%BE%E5%A0%82%E7%AC%94%E8%AE%B0-01-%E5%B1%80%E9%83%A8%E6%9C%80%E5%B0%8F%E5%80%BC%E4%B8%8E%E9%9E%8D%E7%82%B9.md)

## 一手来源

- [项目 README](https://github.com/lucky-Luor/bilibili-douyin-notes#readme)
- [Agent Skill 定义](https://github.com/lucky-Luor/bilibili-douyin-notes/blob/main/SKILL.md)
- [MIT 许可证](https://github.com/lucky-Luor/bilibili-douyin-notes/blob/main/LICENSE)
