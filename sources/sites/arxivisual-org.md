# arXivisual 项目站

> 来源归档（ingest · 项目页核查）

- **标题：** arXivisual — arXiv Papers, Visualized
- **URL：** <https://arxivisual.org>
- **GitHub：** <https://github.com/rajshah6/arXivisual>
- **创建者：** Armaan Gupta、Nikhil Hooda、Raj Shah、Ajith Bondili
- **入库日期：** 2026-09-18
- **一句话说明：** 将任意 arXiv 论文转为 **交互式 scrollytelling** 阅读页：分段摘要、KaTeX 公式与 **AI 生成 Manim 动画 + 语音旁白**；URL 技巧 `arxiv.org` → `arxivisual.org`（在 `arxiv` 后插入 `isual`）。

## 开源核查（2026-09-18）

| 资源 | 状态 | 链接 |
|------|------|------|
| 前后端代码 | **已开源**（可克隆本地运行） | [rajshah6/arXivisual](https://github.com/rajshah6/arXivisual) |
| LICENSE 文件 | **未检测到** | GitHub API / 根目录无 `LICENSE`（截至入库日） |
| 在线服务 | **免费 Web 应用** | [arxivisual.org](https://arxivisual.org) |
| Explore 画廊 | **已处理论文缓存** | 站点 `/explore` |

## 核心能力（站点 / README 对齐）

- **输入：** 粘贴 arXiv URL 或 ID；或把 `arxiv.org/abs/<id>` 改为 `arxivisual.org/abs/<id>`。
- **输出：** 按 section 滚动的阅读体验；嵌入 Manim MP4；Azure OpenAI TTS 旁白（`gpt-4o-mini-tts`，gTTS 回退）。
- **管线：** Ingest → Analyze → Plan → Generate（Manim 代码）→ 四道 Validation → Render → R2/本地存储。
- **观测：** Langfuse 追踪 LLM 调用与每篇成本。

## 关联

- [arxivisual 仓库](../repos/arxivisual.md)
- [arXiv 平台实体](../../wiki/entities/arxiv.md)
- [Manim 实体](../../wiki/entities/manim.md)
