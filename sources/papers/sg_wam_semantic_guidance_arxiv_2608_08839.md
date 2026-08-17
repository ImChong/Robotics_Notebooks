# SG-WAM: Text-Grounded and Spatial-aware Semantic Guidance for World-Action Models（arXiv:2608.08839）

> 来源归档（ingest）

- **标题：** SG-WAM: Text-Grounded and Spatial-aware Semantic Guidance for World-Action Models
- **缩写 / 框架：** **SG-WAM（语义引导）** — **不是** arXiv:2608.01397 的 Self-Guided World Modeling
- **类型：** paper / world-action-models / vlm / semantic-guidance
- **arXiv：** <https://arxiv.org/abs/2608.08839>
- **项目页（文内）：** <https://livfour.github.io/SG-WAM/> — 截至 2026-08-17 **HTTP 404**（归档见 [`sources/sites/livfour-sg-wam.md`](../sites/livfour-sg-wam.md)）
- **作者：** Junjie He、Junfeng Li、Zhide Zhong、Haodong Yan、Ruixin Li、Yangyang Zheng、Jiaguan Zhu、Tianran Zhang、Yuqiao Du、Wen Chen、Shunbo Zhou、Haoang Li
- **机构：** 香港科技大学广州校区（HKUST-GZ）；Ola Dimensions
- **入库日期：** 2026-08-17
- **一句话说明：** 用 VLM 规划器预测 text-grounded 与 spatial-aware semantic foresight，作为高层语义注入 WAM，纠正「指令与视频预测错位」。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** 论文 metadata 写 `https://livfour.github.io/SG-WAM/`，打开为 **404**。同作者组的 [DyPES-VLA](https://livfour.github.io/DyPES-VLA_RELEASE/) 项目页可访问，说明 `livfour` 账号仍在，本页尚未挂上。
- **命名碰撞：** [sg-wam.github.io](https://sg-wam.github.io/) 与 [ReturnZhao/SG-WAM](https://github.com/ReturnZhao/SG-WAM) 属于 **另一篇** *Self-Guided World Modeling in Geometry-Aware Policy Space*（arXiv:2608.01397），README 写 Code coming soon。**禁止合并节点。**
- **结论：** **项目页宣称但 404；训练/推理代码未发布。** 源码运行时序图标 **不适用**。

## 摘录 1：机制

现成 CLIP/T5 文本编码与当前观测解耦，WAM 主要靠视觉线索生成未来，导致语义错位。规划器（Qwen3.5）在图像–指令序列上追加 query token：base 组共享语义，两组特异 token 分别对齐 SigLIP2（text-grounded）与 Depth Anything 3（spatial-aware）。注入视频专家的并行 cross-attention；动作专家经联合注意力继承引导。三阶段：先训规划器 → 与视频专家共训 → 再加动作专家。推理丢掉教师编码器。

## 摘录 2：数字

- LIBERO 平均 **98.7%**（文称超过 LingBot-VA 98.5%；相对 FastWAM / GE-Act +1.1 / +2.2）。
- LIBERO-Plus 平均 **81.3%**（+3.5；语言扰动 81.7%，+2.2）。
- 真机四任务（猕猴桃入篮、叠碗、填锅、提锅；每任务 100 条示教 / 50 trial）全面超过 GE-Act 与 FastWAM；未见高度与光照泛化仍最优。

**对 wiki 的映射：** [`wiki/entities/paper-sg-wam-semantic-guidance.md`](../../wiki/entities/paper-sg-wam-semantic-guidance.md)；交叉 [WAM](../../wiki/concepts/world-action-models.md)、[4D-WAM](../../wiki/entities/paper-4d-wam.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（项目页 404；与 Self-Guided SG-WAM 消歧）
