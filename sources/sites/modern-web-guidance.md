# Modern Web Guidance（Chrome for Developers）官方文档核查

> 一手资料：官方产品文档与公开说明页。

- **标题：** Modern Web Guidance | Chrome for Developers
- **类型：** site
- **链接：** https://developer.chrome.com/docs/modern-web-guidance
- **指南目录：** https://developer.chrome.com/docs/modern-web-guidance/guides
- **代码：** https://github.com/GoogleChrome/modern-web-guidance-src（已开源，Apache-2.0）
- **可安装技能：** https://github.com/GoogleChrome/modern-web-guidance（已开放）
- **源码归档：** [googlechrome-modern-web-guidance.md](../repos/googlechrome-modern-web-guidance.md)
- **wiki 映射：** [Modern Web Guidance](../../wiki/entities/modern-web-guidance.md)
- **核对日期：** 2026-11-16
- **核对范围：** 官网首页的产品定位、安装、兼容的 Coding Agent、实例提示词；官方 guides 分类导航与代表用例；上游 GitHub README、SKILL 与源 CLI 对照。
- **通读：** 已完成

## 核查结论

- **功能：** 为 AI 编码代理提供现代 Web 平台 API、CSS、HTML、无障碍、性能、安全和跨浏览器回退建议；其提供的是指南和工具，**不是一个 Web 浏览器或新建 AI 模型**。
- **安装：** 官网推荐在项目目录运行 `npx modern-web-guidance@latest install`；亦提供 `npx skills add GoogleChrome/modern-web-guidance` 等方式。
- **指南示例：** 原生 `dialog`、popover、CSS Anchor Positioning、container queries、动画、Web 性能等；需要逐项查看兼容性。
- **源码/数据开放：** 技能分发仓与实际源码仓均公开；源仓包含 CLI、静态扫描、指南编译与评测工具。未发现需要下载权重的数据驱动模型，训练数据/权重概念不适用。
- **注意：** 官网页面的教程提示词是使用案例，并不能代替在各浏览器执行兼容性与可访问性测试。
