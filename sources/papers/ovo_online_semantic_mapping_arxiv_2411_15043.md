# Open-Vocabulary Online Semantic Mapping for SLAM（OVO）

> 论文一手资料核对｜具身语义认知地图的一个在线实例级实现，而非该概念的全部定义。

- **标题：** Open-Vocabulary Online Semantic Mapping for SLAM
- **作者：** Tomas Berriel Martins, Martin R. Oswald, Javier Civera
- **版本：** arXiv:2411.15043v3（2025-09-29）
- **原文 HTML：** <https://arxiv.org/html/2411.15043>
- **原文 PDF：** <https://arxiv.org/pdf/2411.15043>
- **源码：** <https://github.com/tberriel/OVO>
- **核对范围：** 已阅读 arXiv v3 正文 Abstract、I–V（动机、方法、3D 实例关联与回环、实验和局限）、评测表，以及附录 A（数据与实现）、B（分割模型、视觉语言编码、描述子融合、速度与 mask bleeding 消融）。
- **通读：** 已完成
- **核对日期：** 2026-11-14
- **复现状态：** 仅核对论文，未在本次任务中执行官方源码或复现实验。

## 基于原文的关键结论

1. **输入与表示：** 带位姿 RGB-D 关键帧（相机位姿可由 ORB-SLAM2 / Gaussian-SLAM 提供）；对齐 2D 分割、3D 点与深度，建立实例级 3D segment，并保存可见度较好的关键帧视角。
2. **语义：** 每个 3D 实例积累视觉语言描述子，查询时编码文本并按 cosine similarity 检索，不要求事先固定完整类别表。论文提出融合全图、蒙版图和局部裁剪描述子的权重网络，而非对所有视角无条件平均。
3. **在线建图与回环：** 语义地图更新和 CLIP 提取排队异步进行；SLAM 回环校正后合并重复 3D 实例，避免位姿漂移产生多个重复对象。
4. **实验：** 在 ScanNetv2 的五个高质量场景子集（HVS）上，使用 GT 位姿的 OVO-mapping 在 ScanNet20 上 mIoU 38.1%；接 Gaussian-SLAM 则为 29.3%。此处是不同位姿假设，不能把 38.1% 当作真实在线 SLAM 的结果（论文表 III）。
5. **局限：** 室外、强光照变化、运动模糊与错误关联尚是挑战；回环融合可能漏合并实例，CLIP 合并网络对训练可见类别有一定偏置。相机跟踪保持视频率，不代表语义映射每帧同步实时完成。

## 与认知地图概念的边界

OVO 证明「可定位 3D 几何 + 可用文本检索的实例语义」可以在线维护并接入 SLAM，但原论文并未提供完整的自然语言任务规划或无条件安全导航系统。具身语义认知地图的“目标搜索/认知接口”还需规划与决策层。

## 对 wiki 的映射

- [具身语义认知地图](../../wiki/concepts/embodied-semantic-cognitive-map.md)
- [OVO 在线语义建图](../../wiki/entities/ovo-semantic-mapping.md)
