# Flickr30K Entities（arXiv:1505.04870 / IJCV 2017）

> 来源归档（ingest）

- **标题：** Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models
- **类型：** paper / dataset / grounding / coreference / benchmark
- **机构：** UIUC（University of Illinois at Urbana-Champaign）；Broad Institute of MIT and Harvard
- **作者：** Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, Svetlana Lazebnik
- **发布时间：** 2016-09-19（arXiv:1505.04870v4）/ IJCV Vol. 123(1), pp. 74–93, 2017
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/1505.04870>
  - 官方代码与标注：<https://github.com/BryanPlummer/flickr30k_entities>
  - 原始图像申请页：<http://hockenmaier.cs.illinois.edu/DenotationGraph/>
- **入库日期：** 2026-11-14
- **核对范围：** 2026-11-14 通读 arXiv:1505.04870v4 正文与实验部分（行 1–1600，涵盖 §1–§6 与实验表格，未逐条核对末尾参考文献）及 `BryanPlummer/flickr30k_entities` 仓库全部代码与说明（`README.md`、`flickr30k_entities_utils.py`、`train.txt`/`val.txt`/`test.txt`）。未解压读取 `annotations.zip` 内部逐条 XML/TXT 样本。
- **通读：** 已完成
- **一句话说明：** 在 Flickr30K（31,783 张图、158,915 条英文描述）基础上通过两阶段 AMT 众包补充 244,035 条跨句共指链与 275,775 个人工标注边界框，奠定短语级图文定位（phrase localization）标准评测。

## 开源与获取状态

- **标注与解析工具：** 官方仓库公开 `annotations.zip`（`Sentences/*.txt` 共指短语文本 + `Annotations/*.xml` PASCAL VOC 风格框标注）、`train.txt`（29,783）、`val.txt`（1,000）、`test.txt`（1,000）、`UNRELATED_CAPTIONS` 及 Python/Matlab 解析脚本。
- **图像本体：** 版权归 Flickr 原作者所有，需通过 UIUC DenotationGraph 页面填表申请，仅限非商业研究与教育用途（遵守 Flickr Terms of Use）。

## 核心技术要素摘录

1. **数据规模与标注结构：**
   - 延续 Young et al. (2014) Flickr30K 的 **31,783 张图像 × 5 句/图 = 158,915 条英文描述**。
   - **244,035 条共指链（coreference chains）**（平均 7.7–7.8 链/图），关联 **513,644 个实体或场景指代短语（mentions）**（平均 16.6 mentions/图）。
   - **275,775 个人工绘制边界框**（平均 8.7 框/图）：59.1% 的共指链对应单个框，20.0% 对应 ≥2 个框（群组/复数实体合并或并存框），20.9% 无对应框（标注为 `scene` 场景或 `nobndbox` 不可见/非视觉指代；`notvisual` 短语在句子文件中使用链 ID `0`）。
   - 划分方式：29,783 训练 / 1,000 验证 / 1,000 测试（固定 ID 列表）。
2. **两阶段众包质量控制管线（Trusted Worker 模型）：**
   - **Stage 1 共指消解：** 从名词汇块提取 mention 并划归 8 个粗类别（`people`, `body parts`, `animals`, `clothing/color`, `instruments`, `vehicles`, `scene`, `other`），由标注员做 Binary Coreference Link 判断，采用复杂度 `O(|M||C|)` 的 "Greedy Plus" 传递性聚类，再经 Coreference Chain Verification 对失败链按中心名词拆链。
   - **Stage 2 边界框标注：** 依次经 Box Requirement（是否需要画框或标 scene/nobndbox）→ Box Drawing → Box Quality → Box Coverage → Box & Chain Merging（在 IoU ≥ 0.8 阈值合并框，`other` 类提高到 0.9）。估计分块误差 <1%、共指误差 <1%、约 8% 图像存在某种标注误差。
3. **短语定位基准与实验结论：**
   - 协议：以 200 个 EdgeBox 候选框评测 Recall@1/5/10（IoU ≥ 0.5 判定命中）。
   - 基线模型：基于 CCA（4096D HGLMM+word2vec 降维表示 18,000D 文本与 Fast R-CNN VGG-19 4096D 区域特征，N=10 样本重采样）+ PASCAL 20 类检测器 + 区域尺寸先验 + 11 类微调颜色分类器。
   - 测试集（ground-truth NP chunks）性能：Recall@1 = **50.89%**，Recall@5 = **71.09%**，Recall@10 = **75.73%**；但 `body parts`（R@1 仅 17.21%）与多人物消歧仍是显著短板，且区域定位增益无法直接大比例转化为整图-句检索（仅约 1–2% 提升）。
4. **官方代码解析逻辑：**
   - `flickr30k_entities_utils.py` 提供 `get_sentence_data(fn)`（解析 `[/EN#<chain_id>/<type...> <phrase>]` 标记为词索引与类型列表）与 `get_annotations(fn)`（`xml.etree.ElementTree` 读取 `<size>`、`<object>`，区分 `<bndbox>` 转换为 0-based `[xmin, ymin, xmax, ymax]`，或记录 `nobox` / `scene` 列表）。

## 对应 wiki 节点

- [Flickr30K Entities](../../wiki/entities/dataset-flickr30k-entities.md)
