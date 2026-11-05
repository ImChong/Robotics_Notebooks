# Mika

- **类型：** repo
- **链接：** <https://github.com/wangzan101/Mika>
- **许可：** Apache-2.0
- **核查版本：** `8d97468a75917a7966792dc486a5632971f26e7e`（0.1.36），2026-11-05
- **开源状态：** Android 应用及 Kotlin agent 核心代码公开；模型需自行配置服务或账号。
- **资料：** [README](https://github.com/wangzan101/Mika/blob/8d97468a75917a7966792dc486a5632971f26e7e/README.md)、[功能说明](https://github.com/wangzan101/Mika/blob/8d97468a75917a7966792dc486a5632971f26e7e/docs/features.en.md)。
- **知识页：** [Mika](../../wiki/entities/mika.md)。

README 介绍后台屏幕操作、浏览器、记忆、模型连接、设备要求和隐私边界；默认 Ask 模式在花钱前确认，发消息等操作直接执行。

**核查范围：** 阅读 README、功能说明和 `AgentLoop.kt`、`ActionExecutor.kt`、`ModelCaller.kt` 的相关路径；未完整追踪 Android 应用入口到手机控制工具的调用链，也未真机运行。源码时序图仅覆盖已核对的 agent 循环；后续需补全手机侧实现。
