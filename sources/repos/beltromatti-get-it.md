# Get It.（beltromatti/get-it）

> 来源归档

- **标题：** Get It.
- **类型：** repo（Electron 桌面学习应用；Next.js、React、TypeScript）
- **作者：** Get It. team（GDG AI Hack Milan 2026 团队）
- **官方仓库：** <https://github.com/beltromatti/get-it>
- **项目页：** <https://getit.noesisai.it/>
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-15
- **一句话说明：** 本地优先的 PDF/Markdown 学习桌面应用：从文档提取概念、生成可选 2D/3D/公式/图表可视化，并用知识图谱、主动回忆、测验、Feynman 讲解和学习日志组织复习。
- **开源状态：** 已开源，代码仓库与项目页均公开；README 提供本地开发、构建及发布说明。
- **沉淀到 wiki：** 是 → [`wiki/entities/get-it-study-companion.md`](../../wiki/entities/get-it-study-companion.md)

## README 与项目页要点（归纳，2026-10-15）

- **文档入口：** 接受可搜索文本 PDF 与 Markdown。扫描件/仅图像文档会在导入检查阶段被拒绝；README 记载上传上限为 150 页。
- **概念可视化：** 概念检测 agent 为页面术语打标签并指定 renderer 类型；右侧可按需生成 Three.js 3D 场景、Canvas 动画、公式、图表或引用来源。项目页展示最多四个视觉任务并行渲染。
- **学习图谱：** 为单份文档创建约 6–25 个概念节点和类型化关系；概念有 memory、comprehension、structure、application 四个 0–100 分数，并标示进展。
- **复习闭环：** 聊天、主动回忆卡片、选择题测验、Feynman 式口头讲解共同写入本地学习日志；每次学习后由 evaluator agent 读取日志并更新概念分数。
- **模型接入：** Electron 桌面应用支持 Codex、Claude、Gemini 官方 CLI/API 接入，也支持 OpenAI-compatible endpoint（README 提到可接本地 Ollama）。用户自行提供已有账号或 API key；没有项目方的模型代理服务。
- **本地与数据流：** 文档、提取缓存、图谱和日志保存在用户设备；模型请求经用户选择的模型供应方处理，不经 Get It. 自有服务器转发。README 表示另有匿名打开/更新 ping（install id、版本、OS），可用 `GETIT_DISABLE_ANALYTICS=1` 关闭。
- **桌面打包：** GitHub Release 提供 macOS、Windows 与 Linux 安装包；仓库发布 workflow 按 tag 构建多平台产物。
- **项目页状态差异：** 当前项目页标注 v1.1.0，仍提示安装包为未签名 hackathon builds；仓库 README 则说明后续 v1.2.1 起 macOS 包已签名并 notarized。以 GitHub Releases 与随版本更新的 README 为准，下载前核对版本和平台签名状态。

## 对 Robotics_Notebooks 的映射

- **主实体：** [`get-it-study-companion.md`](../../wiki/entities/get-it-study-companion.md)
- **相关技术参照：** [Three.js Game Skills](../../wiki/entities/threejs-game-skills.md)（浏览器 3D 工程技能；Get It. 使用 Three.js 作教育可视化，但不是游戏创作工具）
- **项目页：** [`getit-noesisai.md`](../sites/getit-noesisai.md)
