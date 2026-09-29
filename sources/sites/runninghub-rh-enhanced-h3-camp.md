# RunningHub RH Enhanced · MiniMax H3（Camp 项目页）

> 来源归档

- **标题：** RH Enhanced / RunningHub MiniMax H3 Camp 项目（用户指定链）
- **类型：** site（云端 ComfyUI / Camp 工作流入口）
- **URL：** <https://rhtv.runninghub.cn/projects/camp/view/2103108085243867138>
- **workflowId（自 URL 解析）：** `2103108085243867138`（RunningHub [Workflow API](https://www.runninghub.ai/runninghub-api-doc-en/doc-8287463) 以地址栏 ID 调用）
- **平台：** [RunningHub](https://www.runninghub.cn/) / [runninghub.ai](https://www.runninghub.ai/) — ComfyUI 云端执行与 Workflow API
- **入库日期：** 2026-09-29
- **一句话说明：** RunningHub **rhTV Camp** 上的 MiniMax H3 **增强版**工作流入口（用户称 **RH Enhanced**）；与官方插件 [ComfyUI-RH-MiniMax-H3](../repos/comfyui-rh-minimax-h3.md) 同生态，面向 **云端一键跑通** 或 **API 批量任务**（需会员/API Key）。
- **步骤 2.5（页面核查）：** Camp 页为 **SPA**，无登录时公开 HTTP/API（`403 TOKEN_MISSION`）**无法抓取正文**（2026-09-29）；**工程开放度**以 MiniMax 官方 [GitHub/HF](../repos/minimax-h3.md) + RunningHub **Apache-2.0 插件** 为准；Camp 本身为 **托管工作流**，非模型权重仓。

## 可交叉验证的公开说明（同生态，非 ID 一一对应）

RunningHub 社区公开的 **「MiniMax H3 All-In-One Workflow」**（[帖子示例](https://www.runninghub.ai/post/2094673232666742786/)）描述与 **RH Enhanced** 同类诉求，可作 Camp 能力 **策展参照**（入库日未确认与 `2103108085243867138` 字节级同一 JSON）：

- **合并 H3 模型路径：** 同一工作流覆盖 text-to-video、image-to-video、首尾帧、多参考生成。
- **两段式 latent 放大：** 低分辨率快速预览选 seed → 再 refine 到约 **1080p**，节省全分辨率试错成本。
- **质量修复取向：** 缓解 acceleration **LoRA** 的「油腻/塑料」质感、远距人脸糊、高运动场景细节损失。
- **教程外链（帖子）：** <https://youtu.be/RicFavgpL5o>

## API 调用提示

- 文档要求 workflow 在平台 **手动跑通一次** 后再用 API。
- 创建任务：`POST https://www.runninghub.ai/task/openapi/create`，body 含 `apiKey` + `workflowId`（Consumer 会员或 Enterprise 额度）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MiniMax H3 实体](../../wiki/entities/minimax-h3.md) | 模型与开源边界 |
| [ComfyUI-RH 插件](../repos/comfyui-rh-minimax-h3.md) | 本地同栈节点 |
| [ComfyUI](../../wiki/entities/comfyui.md) | 工作流运行时 |
