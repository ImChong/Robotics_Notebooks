# MOSS API 开放平台与模思智能官网

- **类型：** 官方网站 / API 文档
- **官网：** <https://www.moss.cn/>
- **平台：** <https://platform.moss.cn/>
- **文档：** <https://platform.moss.cn/en/docs/getting-started/overview/>
- **模型目录：** <https://platform.moss.cn/en/docs/models/>
- **核查日期：** 2026-11-02；网页动态更新，无固定版本号。
- **维护方：** 上海模思智能科技有限公司。
- **源码与数据：** 官网列出独立模型的开源资源；平台页面未发现平台服务端源码或训练数据链接，不能据此推断平台本身开源。MOSS-Image 的代码与权重见[项目归档](moss-image-project.md)。
- **沉淀到 wiki：** [MOSS API 平台](../../wiki/entities/moss-api-platform.md)。

## 核查范围与摘录

核对官网首页与关于页面，以及 API 平台主页、核心概念、Quickstart、模型列表、API Overview 和实时视频理解指南。平台提供 Playground、密钥管理、用量与价格入口；文档列出语音生成与转写、图像/视频理解、图像生成及实时交互接口。平台与各模型的开源状态需分别判断。

快速开始使用 `moss-tts-1.5-flash` 调用 `POST https://api.mosi.cn/v1/audio/speech`，传入文本和 `voice_id`；成功后按返回类型读取结果。API 文档区分同步、异步和流式；实时视频理解使用 WebSocket。
