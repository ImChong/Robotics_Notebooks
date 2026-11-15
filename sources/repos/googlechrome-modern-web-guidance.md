# Modern Web Guidance（GoogleChrome）官方仓库归档

> 一手资料层：技能分发仓库、技术规则与 CLI 源码核对，不复制上游 250 项指南正文。

- **标题：** Modern Web Guidance
- **类型：** repo / Agent Skill
- **原始仓库：** https://github.com/GoogleChrome/modern-web-guidance
- **实际源码仓库：** https://github.com/GoogleChrome/modern-web-guidance-src
- **官方文档：** https://developer.chrome.com/docs/modern-web-guidance
- **维护：** Google Chrome、Microsoft Edge、Mozilla Firefox、MDN 团队及 Web 开发社区（官方 README 声明）
- **许可：** Apache-2.0（上游 LICENSE / package.json）
- **版本：** 上游 package.json 标记 1.0.0；快照核查日期 2026-11-16，版本后续会变化
- **入库日期：** 2026-11-16
- **wiki 映射：** [Modern Web Guidance](../../wiki/entities/modern-web-guidance.md)
- **核对范围：** 分发仓库 README 全文（定位、Quickstart、范围、兼容性与安装）、`SKILL.md` 全文（search/retrieve/legacy-scan 流程、Baseline 与 fallback 策略）、`package.json`；代表指南 `guides/ui-components/dialog.md` 和 `navigation-drawer.md`；实际源码仓 `src/cli/modern-web.ts`（CLI 分发与查找/读取/扫描入口）、`src` 目录与源码 `package.json`。这是项目主干与两个实例指南的通读，不代表逐篇审阅所有 250 项特性指南。
- **通读：** 已完成

## 项目原始机制

1. **解决问题：** 编码 Agent 容易沿用旧式 HTML/CSS/JS 写法；将专家维护的现代 Web 规范、浏览器支持信息及合理降级策略做成可按需检索的技能指南。
2. **技能入口：** `skills/modern-web-guidance/SKILL.md` 指示 Agent 以 2–5 个英文词搜索单项 UI/Web API 需求，匹配后用 `retrieve` 按 ID 获取指南；仅当搜索无相关结果时才改写查询或浏览列表。
3. **主要命令（README 1.0）：**
   - `npx modern-web-guidance@latest install` — 安装交互向导。
   - `npx modern-web-guidance@latest search "animate a dialog modal backdrop"` — 返回匹配的指南 ID 与说明。
   - `npx modern-web-guidance@latest retrieve "animate-to-from-top-layer"` — 读取完整指南。
   - `npx modern-web-guidance@latest legacy-scan .` — 从存量前端代码检出部分旧式模式，仍需人工确认误报。
4. **机制例子：** `dialog.md` 优先使用原生 `<dialog>` 的模态语义、焦点和 Escape 管理；在 `closedby` 尚未全浏览器支持时，须考虑兼容性与降级，不是所有新 API 都可无条件使用。
5. **分层关系：** `GoogleChrome/modern-web-guidance` 是可安装技能及指南的分发仓库；`GoogleChrome/modern-web-guidance-src` 保存源代码、指南生产、评测和 CLI 实现。源码 `src/cli/modern-web.ts` 路由 `search` 至 `src/rag/search.ts`，`retrieve` 至 `src/rag/retrieve.ts`，`legacy-scan` 至 `src/analyzing/cli.ts`。
6. **指南输出：** `search` 仅打印 ID 与 description 减少上下文占用；`retrieve` 汇集正文，输出过大（9,500 字节阈值）会存入临时文件并提示读文件，而非静默截断。

## 工程边界

- **领域：** 面向 Web 前端最佳实践，不是 ROS / MuJoCo / Isaac / 机器人控制器的专用质量检查工具。
- **Baseline：** `SKILL.md` 指示 Baseline Widely available 可直接采用，其他功能须按对应指南的兼容策略回退；不能把某项指南的浏览器支持视为永久不变。
- **扫描：** `legacy-scan` 只覆盖可静态识别的部分指南（技能文档注明约 20%），不能将零发现解释为代码无兼容性问题。
- **评测：** 上游公开 eval dashboard 与评测源仓；本次未运行 CLI、legacy-scan 或复现实验，不引用未经本轮验证的质量提升百分比。
- **版权：** 只保留事实提炼、命令和源码路径，读者应前往上游读取各完整指南。

## 关联

- [官方项目文档来源](../sites/modern-web-guidance.md)
- [Modern Web Guidance 实体节点](../../wiki/entities/modern-web-guidance.md)
- [Agent Skills（Addy Osmani）](../../wiki/entities/agent-skills-addyosmani.md)
