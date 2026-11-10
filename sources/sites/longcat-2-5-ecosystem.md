# LongCat-2.5 官方 API 与美团发布资料

- **API 文档：** <https://longcat.chat/platform/docs/>
- **美团新闻稿/经营分析案例：** <https://www.meituan.com/news/NN261102280001343>
- **主技术博客：** <https://tech.meituan.com/2026/11/02/Meituan-LongCat-2.5.html>
- **类型：** 官方平台文档与发布资料
- **入库日期：** 2026-11-10
- **核对范围：** 平台 API 快速入门中的调用地址、兼容协议和模型标识；发布页中的发布定位及案例；平台其它细分 API 文档未逐一验证。
- **通读：** 未完成

## 接口与落地路径

1. LongCat 开放平台提供 **OpenAI 兼容** 与 **Anthropic 兼容** 的接口形式，分别参考 `https://api.longcat.chat/openai` 与 `https://api.longcat.chat/anthropic` 的文档示例。
2. 关注 `LongCat-2.5` 模型名称、`/v1/chat/completions` 等真实接口；认证、额度、速率限制依平台最新说明。
3. 美团新闻发布内容与技术博客是不同性质的资料：前者含经营分析/业务应用案例，不能代替技术评测或第三方可复现实验。

## 风险

- 访问服务依赖账户与 API 凭据；本文没有进行需要凭据的实际调用。
- 新闻页面所载业务示例不构成通用的企业 ROI 预测，也不应直接推断模型的机器人控制能力。

**对应 wiki：** [LongCat-2.5](../../wiki/entities/longcat-2-5.md)
