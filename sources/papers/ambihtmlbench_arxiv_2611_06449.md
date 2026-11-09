# AmbiHTMLBench: Benchmarking Image-to-HTML Reconstruction｜图像到可执行 HTML 的结构与视觉保真评测

> 原论文归档（截至 2026-11-09：核查过论文摘要、官方项目说明和代码 README；全文逐节核对尚未完成）

- **类型：** paper
- **原文：** <https://arxiv.org/abs/2611.06449>
- **项目：** <https://ambihtmlbench.tech/>
- **代码：** <https://github.com/theseus-labs-rsi/AmbiHTMLBench>
- **入库日期：** 2026-11-09
- **核对范围：** 论文摘要与方法/评测选段、官方项目页、主仓 README 及相关运行入口；附录和全文尚未逐页完成核对。
- **通读：** 未完成

## 研究问题与核心事实

- **论文：** <https://arxiv.org/abs/2611.06449>
- **任务：** 根据页面图像重建可执行 HTML，保留文字内容、表格、公式与多栏布局。
- **数据：** 1,115 页面、13 类元素、263 个含表格页面，公开 Ground Truth JSON 与图像数据入口。
- **客观：** HTML 元素匹配、结构度量、文本编辑距离和布局图相似度。
- **主观：** 冻结 OLS5 代理、成对 judge 投票、软 Bradley–Terry 和 bootstrap；代理偏好分不直接等于人类评分。

## 方法与评测摘录

- **1,115 张**页面图像，含 **13 类**注释对象；其中 **263 张**涉及表格，评价多区域页面而非纯文本抽取。
- **客观评分**对齐 HTML 解析后的文本块、表格、公式和布局关系；代码中的输出包括 `table_teds`、`table_teds_s`、`formula_cdm`、`text_edit_distance`、`layout_graph_similarity`。
- **主观评分**使用冻结 OLS5 多裁判代理协议，记录五名 judge 的成对偏好，再用软 Bradley–Terry 排序；论文另有人工标定研究。主观排名代表**偏好代理分**，不能视作客观精度或人类正确率。
- 同一模型可能在结构精度和外观偏好上排序不同，不应将两者合成未定义的单一“网页质量”。

## 复现与保留意见

README 提供评分实现和标注 JSON，但主观评测的原始 judge 结果、图像及人工注释需额外供应；主观 replay 本身不重新调用模型或生成新判断。

本归档将已查过的一手材料与尚待全文复核的结论分开；未在本次会话实际运行论文实验，不应把代码入口的存在解释为分数已复现。

## 对 wiki 的映射

- [AmbiHTMLBench 主实体](../../wiki/entities/paper-ambihtmlbench.md)
- [官方站点归档](../sites/ambihtmlbench.md)
- [代码仓归档](../repos/ambihtmlbench.md)
