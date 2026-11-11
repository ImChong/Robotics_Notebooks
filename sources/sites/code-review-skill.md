# Code Review Skill 官方主页

> 官方项目展示页归档；并非独立新模型或独立源码

- **标题：** Code Review Skill 官方主页
- **类型：** site
- **链接：** https://awesome-skills.github.io/code-review-skill/
- **代码：** https://github.com/awesome-skills/code-review-skill（已开源，MIT）
- **源码归档：** [code-review-skill](../repos/code-review-skill.md)
- **wiki 映射：** [Code Review Skill](../../wiki/entities/code-review-skill.md)
- **检查日期：** 2026-11-11
- **核对范围：** 官方 GitHub Pages 的中英介绍、技术栈、四阶段审查、安装命令、仓库链接和许可说明；与仓库 README / SKILL.md 对照。
- **通读：** 已完成

## 站点核验

- 四阶段 PR 评审：上下文收集 → 架构/性能/测试策略 → 逐行审查 → 总结结论。
- 安装入口为 `npx skills add awesome-skills/code-review-skill`；主页还给出 `--skill code-review-skill` 形式。
- 文本规约依赖 `SKILL.md` 与 `reference/*.md`；展示“20+ 语言 / 21,000+ 行”为项目方维护的统计值。
- MIT 源码仓已公开；项目站没有提供独立权重、训练数据或云托管评审服务。
- 具体脚本行为应以 [pr-analyzer.py](https://github.com/awesome-skills/code-review-skill/blob/main/scripts/pr-analyzer.py) 为准，不能把宣传性“生产级”直接推导成已验证的质量提升。
