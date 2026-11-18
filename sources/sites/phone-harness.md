# Phone Harness 官方主页与开发文档

- **类型：** 官方产品与文档站点
- **官网：** https://phone-harness.com/
- **开发文档：** https://phone-harness.com/docs/introduction
- **代码：** https://github.com/phone-harness/phone-harness
- **归档日期：** 2026-11-18
- **核对范围：** 官方首页的功能/连接模式/托管服务说明，以及官方 Docs Overview（认证、Base URL、会话生命周期和定价引用）；未通读完整 API Reference，未实测云租用。
- **通读：** 已完成
- **源码归档：** [phone-harness 仓库](../repos/phone-harness.md)
- **知识实体：** [Phone Harness](../../wiki/entities/phone-harness.md)

## 页面说明

- 官网明确定位为让 AI Agent 操作 iPhone/Android 的交互控制工具；支持点击、文本输入、滑动和屏幕读取。
- 本地 iPhone 经 iPhone Mirroring；本地 Android 经 ADB。官方另提供云端 Android 模拟器及真实 iPhone 服务。
- 文档的云 API 基础 URL 为 `https://api.phone-harness.com`，使用 Bearer API Key；典型生命周期：创建 session → 等待 ready → 使用 adb / Agent / MCP → 结束 session。
- 官网与文档介绍付费云端托管，**并不意味着所有云服务后端源码均随开源仓库公开**。仓库公开 MIT 授权 CLI 和设备驱动代码，服务侧功能以官网套餐与 API 为准，核验时不对未审计代码作开源断言。

## 工程参考

手机 UI 适合做跨 App Agent 自动化测试、终端数据采集或上层应用交互实验；与机器人真机闭环控制的频率、安全和执行器接口不同。

## 原始资料

- [官网](https://phone-harness.com/)
- [Overview](https://phone-harness.com/docs/introduction)
- [GitHub 源码](https://github.com/phone-harness/phone-harness)
