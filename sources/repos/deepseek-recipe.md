# DeepSeek-AI/deepseek-recipe

> 来源归档（repo）

- **标题：** deepseek-recipe
- **代码：** <https://github.com/deepseek-ai/deepseek-recipe>
- **类型：** Rust 库与 Python 绑定（模型协议适配、提示词编码、响应解析）
- **默认分支：** `main`
- **源码状态：** 已公开
- **许可证：** MIT
- **核验日期：** 2026-10-20

## 一句话摘要

deepseek-recipe 将不同 API 格式的请求转换为统一的 Conversation，渲染 DeepSeek V4/V4.1 提示词并解析模型输出，供上层服务连接推理后端。

## 核心能力

仓库支持 Messages、Chat Completions 与 Responses 请求/响应格式，覆盖文本、图像、思考模式、客户端工具调用、结构化输出、流式响应及输出解析。组件包括 Rust workspace、Python 绑定、提示词编码与解码示例，以及带 mock inference 的 Axum / FastAPI 示例服务。

仓库明确将模型推理、工具执行与 HTTP 传输留给调用方；它本身不是推理引擎或完整服务端。Python 包要求 Python 3.10+，Rust 示例按 README 使用 Cargo 依赖。

## 对机器人系统的参考价值

它提供了一个把上层应用协议、模型提示词和推理后端解耦的实现参考，可用于理解具身模型服务中的请求适配与结构化动作/工具输出接口。仓库针对 DeepSeek V4/V4.1 设计，未提供机器人控制、动作安全校验或机器人专用 API。

## 工程注意事项

图像组件支持 base64 与外部 URL，并会为 DeepSeek V4.1 使用 OpenCV 预处理。README 特别提醒：默认的 `ReqwestImageFetcher` 可连接任意地址；若服务接收不可信请求，应安装 `DestinationFilter`，限制 URL 抓取目标。

## 公开资源

| 资源 | URL |
|------|-----|
| GitHub 仓库 | <https://github.com/deepseek-ai/deepseek-recipe> |
| README | <https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.md> |
| 流式响应说明 | <https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/streaming.md> |
| Tokenizer 使用说明 | <https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/tokenizer.md> |

## 参考来源

- [deepseek-recipe 官方 README](https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.md)
- [GitHub 仓库元数据](https://api.github.com/repos/deepseek-ai/deepseek-recipe)
