# Precise Top-Layer Fabric Segmentation for Fabric Destacking（arXiv:2608.10648）

> 来源归档（ingest）

- **标题：** Precise Top-Layer Fabric Segmentation for Fabric Destacking with Edge- and Shape-Aware Deep Networks
- **类型：** paper / cv / fabric / destacking / segmentation
- **arXiv：** <https://arxiv.org/abs/2608.10648>
- **会议：** IEEE ICMA 2025（作者接受稿）
- **代码：** <https://github.com/bhattner143/top-layer-fab-seg>（归档见 [`sources/repos/top-layer-fab-seg.md`](../repos/top-layer-fab-seg.md)）
- **作者：** Wenbo Dong、Dipankar Bhattacharya、Akinari Kobayashi、Akira Seino、Fuyuki Tokuda、Xuzhao Huang、Kai Tang、Norman C. Tien、Kazuhiro Kosuge
- **机构：** 香港大学（HKU）；大阪大学（Osaka）
- **入库日期：** 2026-08-18
- **一句话说明：** 布料分拣要精确分割最上层；在 encoder-decoder 上加 edge-aware 与 shape-aware（CAD 参考 mask）两路监督。

## 开源状态（步骤 2.5）

- **无独立项目页**；论文 Comments 给 GitHub。
- **仓库核查（2026-08-18）：** [bhattner143/top-layer-fab-seg](https://github.com/bhattner143/top-layer-fab-seg) **空仓**（`size: 0`，contents 404）。
- **结论：** **宣称开源 / 仓为空**；源码运行时序图不适用。

## 摘录

层间边界细、外观相似，纯语义或纯边缘分割常失败。训练架构：主干 encoder-decoder + 边缘分支强化边界 + 形状分支用 CAD mask 对齐整体外形。真机布料数据上优于既有基线（摘要未给具体 IoU 数字）。

**对 wiki 的映射：** [`wiki/entities/paper-top-layer-fabric-seg.md`](../../wiki/entities/paper-top-layer-fabric-seg.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（空仓）
