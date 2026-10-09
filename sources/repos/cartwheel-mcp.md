# Cartwhl/cartwheel-mcp

> 来源归档（ingest）

- **仓库：** <https://github.com/Cartwhl/cartwheel-mcp>
- **项目官网：** <https://getcartwheel.com/>
- **许可证：** MIT（GitHub 仓库元数据与 LICENSE）
- **语言 / 运行时：** JavaScript / Node.js 22+
- **类型：** 本地 MCP 服务；Cartwheel API 接入与 Blender 示例
- **官方关系：** Cartwheel 产品页将 MCP 列为官方接入方式；该仓库由 Cartwhl 组织维护。
- **详情节点：** [Cartwheel Comic](../../wiki/entities/cartwheel-comic.md)
- **入库日期：** 2026-10-09

## 仓库作用

此仓库将 Cartwheel 的公开 API 包装为供 MCP 客户端使用的本地 stdio 服务。README 包含 Comic 视频动捕调用 `generate_motion_from_video`，可捕获 1–4 位人物及可选面部动作；还提供生成/编辑动作、角色创建、场景管理、导出和运动分析工具。视频任务异步运行，客户端需要提交任务并轮询批次结果。

## 快速接入

README 要求 Node.js 22 或更新版本、可访问 API 的 Cartwheel 账号/套餐、项目 API key，以及能启动本地 stdio server 的 MCP 客户端。最小运行路径：

```bash
git clone https://github.com/Cartwhl/cartwheel-mcp.git
cd cartwheel-mcp
npm ci
```

配置 MCP 客户端启动 `node /absolute/path/to/cartwheel-mcp/src/index.mjs`，并通过环境变量提供 `CARTWHEEL_API_KEY`。Blender 只在需要运行示例渲染时才是依赖；MCP 服务本身不要求安装 Blender。

## Comic 视频动捕路径

```text
create_media_upload → 客户端上传视频 → generate_motion_from_video
→ get_batch 轮询 → list_batch_motions 取预览/下载 → 导入 DCC
```

视频动捕会消耗 Cartwheel credits。MCP 的开源范围是调用、工作流与客户端代码，并不包含 Comic 模型权重、训练数据或离线推理实现；使用仍依赖 Cartwheel 服务、API 权限和当期套餐。

## 关联资料

- Cartwheel 产品与模型页：[sources/sites/cartwheel-comic.md](../sites/cartwheel-comic.md)
- 项目实体：[wiki/entities/cartwheel-comic.md](../../wiki/entities/cartwheel-comic.md)
- [MCP 仓库 README](https://github.com/Cartwhl/cartwheel-mcp/blob/main/README.md)
- [MCP 仓库许可证](https://github.com/Cartwhl/cartwheel-mcp/blob/main/LICENSE)
