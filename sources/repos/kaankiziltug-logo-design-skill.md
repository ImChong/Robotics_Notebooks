# Logo Design Skill（kaankiziltug/logo-design-skill）

> 来源归档

- **标题：** Logo Design Skill
- **类型：** repo（Agent Skill + SVG 设计与检查工具）
- **作者：** Kaan Kızıltuğ
- **链接：** https://github.com/kaankiziltug/logo-design-skill
- **适用代理：** Claude、Gemini CLI、Codex、Cursor、GitHub Copilot 等支持 Agent Skills 的代理
- **代码：** https://github.com/kaankiziltug/logo-design-skill（**已开源**；技能文本、脚本、模板与目录数据为 MIT）
- **入库日期：** 2026-10-15
- **一句话说明：** 面向从 brief 到可交付 SVG 的完整 Logo 设计 skill，提供逐步确认的概念流程、依赖轻量的 SVG 检查/展示脚本与 1,400 多个真实品牌 Logo 的分类参考库。
- **为什么值得保留：** 它将品牌设计原则转成代理可执行的过程和 SVG 检查工具，并为机器人产品、实验室与开源项目提供一套可复用的视觉身份设计参考；与机器人控制和建模工具无直接集成。
- **开源状态：** 已开源。README 同时说明 `assets/library/svg/` 内第三方品牌 Logo 属于其各自商标，仅用于参考与教育，不受仓库 MIT 许可覆盖。
- **沉淀到 wiki：** 是 → [`wiki/entities/logo-design-skill.md`](../../wiki/entities/logo-design-skill.md)

## README 要点（归纳，2026-10-15）

- **工作流：** 整理 brief 与假设 → 检索行业惯例 → 生成 8–12 个概念方向 → 为其中 3 个构建 SVG → 运行尺寸/单色/反白等检查 → 展示总览与推荐，并停在用户选择检查点；用户选定方向后才继续制作完整品牌套件。
- **设计知识：** 覆盖发现与 brief、标志类型、几何构成与光学校正、色彩、字体、组合锁定、测试、演示、交付、改版与视觉识别系统。
- **参考库：** 1,400 多个真实品牌 SVG，按标志类型、技术、几何、主题、字体、气质和行业分类；README 强调用于研究与避免雷同，不得直接复制。
- **本地工具：** Python 脚本可搜索参考库、审查 SVG 结构与复杂度、制作概念/测试页和客户演示板、渲染 PNG，并导出 favicon、App 图标与 Web 图标集。
- **兼容性：** 技能以开放 Agent Skills 格式组织；Python 工具主要使用标准库。README 指出 SVG 位图导出可使用 CairoSVG、librsvg、Inkscape、Chromium 等本机渲染器。
- **边界：** README 中的 1,400+ 参考 SVG 是第三方商标素材；MIT 只覆盖技能文本、脚本、模板和目录数据。商用交付仍需另行进行商标、近似标志与字体许可核查。
