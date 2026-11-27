# Google Cloud · Gemini 蒸馏服务

- **类型：** site（官方技术文档）
- **机构：** Google Cloud（谷歌）
- **链接：** [简体中文官方服务指南](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/distillation?hl=zh-cn)
- **文档版本：** 主指南页脚更新于 2026-09-16 UTC；本条目核对于 2026-11-27。
- **沉淀到 wiki：** [Gemini 蒸馏服务](../../wiki/entities/google-gemini-distillation-service.md)
- **通读：** 已完成
- **核对范围：** 中文指南全文（访问、模型、数据、作业创建与续训、监控与取消、端点评估、限制和配额）；对照发布记录、v1beta1 REST 通用资源定义与示例字段。服务端实现及私有访问环境不在核查范围内。

## 原始资料入口

- [Google Cloud 发布记录](https://docs.cloud.google.com/gemini-enterprise-agent-platform/release-notes)：2026-05-08 条目公告 Early Access。
- [v1beta1 tuningJobs REST 参考](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1beta1/projects.locations.tuningJobs#DistillationSpec)：作业、学生来源、蒸馏字段和输出资源。通用 schema 并非 Early Access 全功能清单。
- 主指南已有 REST 客户端示例；未找到服务端实现仓库。本次仅做公开文档及本地输入格式核验，未访问项目、提交训练、上传数据或调用部署端点。
