# Gitingest 在线入口（gitingest.com）

> 来源归档（ingest）

- **类型：** site / hosted developer tool
- **URL：** <https://gitingest.com/>
- **代码：** <https://github.com/coderamp-labs/gitingest> — [仓库归档](../repos/coderamp-labs-gitingest.md)
- **入库日期：** 2026-10-08
- **一句话说明：** 将 GitHub 仓库转为可供 LLM 阅读的文本 digest；可通过将 GitHub URL 中的 hub 替换为 ingest 快速打开对应页面。

## 页面与产品要点

- 输入公开仓库 URL，在线生成 prompt-friendly digest；官方 README 亦记录 CLI、Python 包及可自托管服务。
- 输出核心是摘要、文件树、文件内容和 token/大小统计，用户可据此判断上下文规模并复制给模型。
- 官方 README 链接 Chrome、Firefox、Edge 浏览器扩展；扩展属于独立仓库/分发面。
- 私有仓库访问需要 GitHub token；本归档不推断托管站点的保留、训练或访问控制承诺。

## 数据与使用边界

本地 CLI / Python 处理与托管网站处理不是同一条信任边界。将私有或敏感代码交给托管服务前，应查阅当前隐私与服务条款；对敏感仓库优先在自有环境使用本地接口或自托管部署，并限制纳入文件。

## 关联资料

- 仓库归档：[coderamp-labs/gitingest](../repos/coderamp-labs-gitingest.md)
- Wiki：[Gitingest 实体页](../../wiki/entities/gitingest.md)