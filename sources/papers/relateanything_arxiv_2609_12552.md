# RelateAnything（arXiv:2609.12552）

> 来源归档（paper）

- **标题：** RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs
- **类型：** paper
- **作者：** Maëlic Neau（Independent Researcher）
- **arXiv：** <https://arxiv.org/abs/2609.12552>
- **PDF：** <https://arxiv.org/pdf/2609.12552>
- **项目页：** <https://maelic.github.io/RelateAnythingProject/>
- **代码：** <https://github.com/Maelic/RelateAnything>
- **模型：** <https://huggingface.co/collections/maelic/relateanything>
- **数据集：** <https://huggingface.co/datasets/maelic/RA-4M>
- **入库日期：** 2026-09-21
- **一句话说明：** 53M 实时开放词汇关系预测：像素 + 任意区域源 + 推理时谓词字符串表 → 打分三元组；配套 RA-4M 语料与 OV-SGG-Bench 六轴评测。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-21）：Apache-2.0 代码、HF 权重（relsgg-vits16/16+/vitb16）、RA-4M 与 OV-SGG-Bench 数据集、浏览器 ONNX demo。

## 核心摘录

1. **解耦输入：** 关系头 **从不** 接收物体类别标签；区域可来自任意检测器、类无关分割器或 GT box。
2. **开放词汇：** 谓词为推理时提供的字符串表，经蒸馏 text encoder 编码为 ℓ₂ 归一化 bank；换词汇 = 换矩阵行，无需重训。
3. **RA-4M：** 474k 图、4.3M 关系、10,102 自由文本谓词；Gemma 4 (26B) + SAM 2.1 三 pass 标注 + **确定性几何 gate**（拒绝 11.3%）。
4. **训练难点：** 万级词汇下监督为 positive-unlabeled；对比 text encoder 反义词 cosine≈0.95 → 蒸馏 + antonym-repulsion。
5. **OV-SGG-Bench：** 六轴（transfer / precision / open vocab / deployment / graph quality / spatial）跨数据集评分；composite 40.1 vs OvSGTR 11.8。
6. **性能：** A40 上 20 ms/frame；cross-dataset mean recall 为 OvSGTR 同级方法 **2.3–3.5×**，稀有谓词 **5–21×**。

**对 wiki 的映射**

- [paper-relateanything](../../wiki/entities/paper-relateanything.md)
- [relateanything-project](../sites/relateanything-project.md)
- [Maelic/RelateAnything](../repos/maelic-relateanything.md)
