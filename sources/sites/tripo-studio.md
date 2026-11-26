---
type: site
url: https://studio.tripo3d.ai/
title: "Tripo Studio — AI 3D 工作台官方文档"
added: 2026-11-26
---

# Tripo Studio 官方资料归档

- **产品：** Tripo Studio，由 [三启万物（VAST）](https://www.vastworld.ai/zh/about) 构建；条款所列服务方为 Holymolly。
- **入口：** [Studio](https://studio.tripo3d.ai/) · [Creator Hub](https://studio.tripo3d.ai/creator-hub)。
- **版本：** 产品文档标注 2026 年 10 月；H3.5 Preview 发布说明更新于 2026-11-25；用户协议更新于 2026-11-13。查阅于 2026-11-26。
- **核对范围：** 通读所列核心产品文档、H3.5 Preview 发布说明及用户协议有关条款，覆盖生成输入、H/P 模型、网格编辑、绑定、导出、Canvas 节点和共享、积分及内容许可；不包含登录后生成、实际下载或打印。
- **通读：** 已完成

| 主题 | 官方一手资料与记录 |
|---|---|
| 产品与输入 | [产品介绍](https://www.tripo3d.ai/product-doc/getting-started/what-is-tripo-studio) · [Image to 3D](https://www.tripo3d.ai/product-doc/model-generation/image-to-3d)。可由文字、单张或多角度参考图生成，也能上传既有网格做后处理；单图背面为推断。 |
| 模型系列 | [H Series](https://www.tripo3d.ai/product-doc/ai-3d-models/h-series-hd-model) · [P Series](https://www.tripo3d.ai/product-doc/ai-3d-models/p-series-smart-mesh) · [H3.5 Preview 发布](https://www.tripo3d.ai/blog/tripo-h3-5)。H 偏细节；P 偏拓扑与面数，纹理后做。H3.5 Preview 是三角网格，最高 500 万面；P2.0 支持最多 5 万三角面或 2.5 万四边面。 |
| 网格处理 | [Retopo](https://www.tripo3d.ai/product-doc/module-post-processing/retopo) · [Smart UV](https://www.tripo3d.ai/product-doc/module-post-processing/smart-uv) · [纹理](https://www.tripo3d.ai/product-doc/module-post-processing/texture-pbr) · [分件](https://www.tripo3d.ai/product-doc/module-post-processing/segmentation)。UV 重展会丢弃旧纹理；分件后检查边界和闭合。 |
| 绑定与交付 | [Rigging & Animation](https://www.tripo3d.ai/product-doc/module-post-processing/rigging-animation) · [Export & DCC](https://www.tripo3d.ai/product-doc/export-and-integration/export-dcc)。FBX 可保留骨骼和动画，GLB 适合轻量预览，STL/3MF 用于切片；以对应导出面板为准。 |
| Canvas | [Agentic Canvas](https://www.tripo3d.ai/product-doc/agentic-canvas/agentic-canvas-guide)。节点输出可接到下游节点；项目分享与素材公开状态分开，复制工作流不会解锁私有资产；仍在逐步开放。 |
| 积分与下载 | [定价](https://www.tripo3d.ai/pricing) · [Pricing & Credits](https://www.tripo3d.ai/product-doc/getting-started/pricing-credits)。Free 标注每月 200 积分；H3.5 Preview 可消耗积分生成/查看，导出需付费。定价页和 11 月发布说明将 H2.5 免费导出列为 15 件/月，10 月 H Series 与 Export & DCC 文档写 25，实际以账户当期限制为准。 |
| 内容授权 | [2026-11-13 用户协议](https://www.tripo3d.ai/terms) §3.7(f)、§5.2–5.4、§6.4。免费用户产物默认公开，有权处置的部分适用广泛公共许可；公开付费产物对他人通常仅限个人非商用；活动须核对其规则。定价表、积分文档和[商用帮助](https://www.tripo3d.ai/help/privacy-policy/how-to-use-tripo-models-commercially)与协议不一致，不能用简写给任意资产授权。 |

- **开放范围：** 未查到 Studio 本体或 H3.5 Preview / P2.0 后端的官方可运行发布；[Python SDK](../repos/tripo-python-sdk.md) 是独立 API 客户端，Web 与 API 不共用额度。TripoSR 是独立研究项目，不视为 Studio 源码。
- **Wiki 映射：** [Tripo Studio](../../wiki/entities/tripo-studio.md)。
