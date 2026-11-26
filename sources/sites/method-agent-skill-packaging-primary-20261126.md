# Agent Skills：Specification

- **URL：** <https://agentskills.io/specification>
- **类型：** 官方文档或作者工程文章
- **核验日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 目录结构、SKILL.md YAML 元数据字段、Markdown 正文、可选目录、逐步加载、相对路径和格式校验。

## 来源摘要

一个技能目录至少包含 SKILL.md，其 YAML 元数据中 name、description 必填。scripts、references、assets 是可选约定；详细说明与辅助文件可以分层加载。allowed-tools 是实验字段，其支持可随实现而异；规范的结构校验不能替代对执行权限和脚本行为的检查。

关联：[方法页](../../wiki/methods/agent-skill-packaging.md)。
