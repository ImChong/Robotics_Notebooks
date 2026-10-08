# GitDiagram 在线应用（gitdiagram.com）

> 来源归档（ingest）

- **类型：** site / hosted codebase explainer
- **URL：** <https://gitdiagram.com/>
- **代码：** <https://github.com/ahmedkhaleel2004/gitdiagram> — [仓库归档](../repos/ahmedkhaleel2004-gitdiagram.md)
- **MCP：** <https://gitdiagram.com/mcp>
- **入库日期：** 2026-10-08
- **一句话说明：** 将 GitHub 代码库转成可交互架构图与流式解释，并可通过 MCP 向 AI agent 提供架构信息。

## 页面与产品要点

- 把 GitHub URL 的 hub 替换为 diagram，或在站点提交仓库地址。
- 交互图中的组件可跳转到相应源文件/目录；可导出 PNG 或复制 Mermaid 源码。
- 支持用 GitHub token 访问私有仓库；token 使用意味着代码会经过所选服务路径，敏感仓库应先评估信任边界。
- 可为仓库观看约一分钟的讲解视频，支持横屏/竖屏 MP4 下载和字幕；截至入库日，新视频生成开放 early access。
- README 声明 MCP 服务可在无需 key/登录的情况下供 agent 使用；能力包括读取公开仓库解释、节点关系与 Mermaid，搜索已存图以及视频信息。

## 使用边界

交互架构图是一种模型辅助理解视图，不应直接当作真实运行拓扑或完整依赖分析。页面与 MCP 面向公开仓库的能力，与私有仓库令牌流程不同；归档不对托管服务数据保留策略作无来源推断。

## 关联资料

- 仓库归档：[ahmedkhaleel2004/gitdiagram](../repos/ahmedkhaleel2004-gitdiagram.md)
- Wiki：[GitDiagram 实体页](../../wiki/entities/gitdiagram.md)