# Cartwheel Comic 产品与模型页

> 来源归档（ingest）

- **项目：** Cartwheel Comic（Cartwheel 的人体运动感知与量化模型）
- **官方模型页：** <https://getcartwheel.com/model/comic>
- **Comic 4.2 发布说明：** <https://x.com/getcartwheel/status/2084371281114443892>
- **Performance Capture：** <https://getcartwheel.com/product/performance-capture>
- **API：** <https://api-docs.getcartwheel.com/api/motion-from-video/>
- **插件页：** <https://getcartwheel.com/plugins>
- **插件条款：** <https://getcartwheel.com/plugin-addendum>
- **官方 MCP 接入仓库：** <https://github.com/Cartwhl/cartwheel-mcp>；归档见 [cartwheel-mcp 源码资料](../repos/cartwheel-mcp.md)
- **入库日期：** 2026-10-09
- **详情节点：** [Cartwheel Comic](../../wiki/entities/cartwheel-comic.md)

## 产品定位与能力

Cartwheel 将 Comic 定位为从单目视频估计人体动作与场景运动信息的模型系列。官方页面列出的估计项包括深度、人体姿态、人体尺度、相机位置、重力、地面接触，以及由此得到的位置、速度和加速度。当前产品页列出 Comic 4、Comic 3 与研究中的 Comic Realtime；Comic 4 支持多人物、面部、尺度与相机信息。

Cartwheel 官方账号发布 Comic 4.2 更新时，宣称改进脚部物理估计，包括亚毫米级脚部落点、脚趾滚动、平底鞋/靴子/高跟鞋和最多四人捕捉。这是产品发布声明。Comic 模型页另列出历史 EMDB 基准：Comic 4 的 foot-skate 数值为 4.41 mm；页面提示该历史结果早于后续 MHR 审核修正。该数值与 4.2 发布声明的“落点精度”并非同一版本或同一指标，不能直接当作相互验证。

API 的视频动捕接口使用 `comicModel: "comic4"`；`footPlanting` 默认开启，会对 Comic 4 输出应用接触感知的足部落地与滑步修正。API 文档支持 1–4 人、可选面部捕捉及 FBX、GLTF、USD、BVH 等输出选项。

## 浏览器、插件与接入边界（2026-10-09 核查）

- Performance Capture 是浏览器产品入口，官方规格列出单条视频不超过 250 MB，可批量处理最多 100 个片段；捕捉数据含 77 个关节/骨骼的位置、深度、相机位置、尺度、地面接触和面部混合形状。
- 官方插件页当前列出 **Windows / Unreal Engine 5.6** 下载与文档入口；Maya、Unity、Blender 均显示 “Notify me”。虽然 Performance Capture 页面列出这些 DCC/游戏工具以及通用动画交换格式，仍需区分插件是否已发布与文件是否可导入。
- 插件和 API 使用 Cartwheel API key，且是否包含访问权限取决于套餐。官方 Cartwheel MCP 是本地运行的 API 客户端；模型权重与推理实现没有随这个 MCP 仓库开放。
- 插件条款说明用户提交内容不会用于训练或改进模型，除非另有书面约定；上传前仍应确认视频人物肖像与数据处理授权。

## 来源入口

- [Comic 官方模型页](https://getcartwheel.com/model/comic)
- [Cartwheel 官方 Comic 4.2 发布说明](https://x.com/getcartwheel/status/2084371281114443892)
- [Performance Capture 产品规格](https://getcartwheel.com/product/performance-capture)
- [API 视频动捕文档](https://api-docs.getcartwheel.com/api/motion-from-video/)
- [3D 插件与平台可用状态](https://getcartwheel.com/plugins)
- [Cartwheel 插件条款](https://getcartwheel.com/plugin-addendum)
