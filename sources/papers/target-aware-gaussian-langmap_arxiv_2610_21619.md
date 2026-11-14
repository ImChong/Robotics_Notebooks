# Training-Free Target-Aware Refinement of 3D Gaussian Language Map

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** Training-Free Target-Aware Refinement of 3D Gaussian Language Map
- **中文：** 目标感知的 3D Gaussian 语言地图免训练细化
- **arXiv：** 2610.21619
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已直接阅读 arXiv HTML 摘要、方法与部分实验/局限章节；尚未逐行检查全文及全部附录，后续继续核对
- **通读：** 未完成
- **论文：** https://arxiv.org/abs/2610.21619
- **HTML：** https://arxiv.org/html/2610.21619
- **PDF：** https://arxiv.org/pdf/2610.21619
- **官方项目页：** https://lab-of-ai-and-robotics.github.io/target-aware-langmap/
- **仓库：** https://github.com/Lab-of-AI-and-Robotics/target-aware-langmap

## 论文机制

从已有场景视图获得文本可提示分割和类无关区域；仅在指定目标的视觉特征中按闭式强度混入文本向量，再由地图蒸馏保留开放词汇基底，目标检索时中心化去掉公共方向偏差。

## 实验与证据边界

在 Replica、ScanNet 与 HM3D-Semantics 的语言目标定位上进行检验；项目页报告 Replica 细化后的 mIoU 约为原基底的两倍。仍须完成基础地图构建和目标区域分割，方法本身免训练不等于全系统免训练。

## 复现与开放状态

已建立公开项目仓库，但 README 明确 Code will be released soon；目前无可运行算法实现。

## 局限

依赖基础地图、开放词汇视觉嵌入及目标掩码质量；公开仓库 README 标注 Code will be released soon，目前仅是项目说明仓库。

## 对 wiki 的映射

- [目标感知的 3D Gaussian 语言地图免训练细化](../../wiki/entities/paper-target-aware-gaussian-langmap.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.21619)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
