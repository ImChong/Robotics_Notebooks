# Magpie 官网（usemagpie.ai）

> 官方项目页归档（2026-11-18）

- **标题：** magpie — Every agent's model. One place.
- **网站：** https://usemagpie.ai/
- **类型：** site（官网 / 文档 / 安装入口）
- **组织：** yetone / Magpie 项目
- **代码：** https://github.com/yetone/magpie
- **文档：** https://usemagpie.ai/docs/start
- **核对范围：** 查看官网首页的 Agent/Provider 支持范围、工作方式、本地网关协议、账号路由、隐私遮蔽、终端命令及官方 GitHub/版本入口；README 另归档在 [仓库一手资料](../repos/yetone-magpie.md)。
- **通读：** 已完成

## 源码及数据开放核查（步骤 2.5）

- **源码已开放：** 官网导航直链 yetone/magpie；包含 Go 主程序、Gateway、Agent 适配器及相关文档，MIT 许可。
- **数据/模型：** 工具集成外部订阅、API Key 和本地模型，**不直接发布通用 LLM 权重**；用户需使用自己有权访问的服务。
- **运行边界：** Mac/Windows/Linux、TUI/CLI 等，官网展示默认本地网关 `127.0.0.1:3425`。
- **安全：** 隐私遮蔽为可选启用功能，非默认启用；远程共享需专门审查认证、网络和凭据策略。

## Wiki 映射

- [Magpie 独立详情节点](../../wiki/entities/magpie.md)
- [官方仓库归档](../repos/yetone-magpie.md)
