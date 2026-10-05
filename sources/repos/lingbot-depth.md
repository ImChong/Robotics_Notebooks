# LingBot-Depth

> 来源归档（国内具身开源全景）

- **标题：** LingBot-Depth
- **类型：** repo
- **机构：** 蚂蚁灵波
- **链接：** https://github.com/Robbyant/LingBot-Depth
- **分类：** 工程与工具
- **入库日期：** 2026-09-06
- **一句话说明：** 蚂蚁灵波 开源项目 LingBot-Depth（工程与工具），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-lingbot-depth.md`](../../wiki/entities/cn-os-lingbot-depth.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-lingbot-depth.md](../../wiki/entities/cn-os-lingbot-depth.md)

## 官方资源补核（2026-10-05）

- **项目页：** <https://technology.robbyant.com/lingbot-depth/>；代码 <https://github.com/robbyant/lingbot-depth>。
- **模型入口：** `mdm.model.v2.MDMModel.from_pretrained`、`python example.py`，输入 RGB、深度与内参，输出补全深度/点。
- **数据：** README / 项目公开约 3,019,200 RGB-D 样本（2026-03-31 公布），并区分真实、VLA、合成与验证部分。
- **工程检查：** 毫米转米与内参归一化须按示例处理；不要把后续 Depth 2 教师扩展视作本仓完整发布。
