# AmbiHTMLBench｜官方项目页

> 来源归档（官方站点）

- **标题：** AmbiHTMLBench: A Benchmark for HTML-based Visually Rich Document parsing
- **类型：** site
- **链接：** <https://ambihtmlbench.tech/>
- **论文：** <https://arxiv.org/abs/2611.06449>
- **代码：** <https://github.com/theseus-labs-rsi/AmbiHTMLBench>（已公开，范围见代码归档）
- **数据：** <https://huggingface.co/datasets/Theseus-Lab/AmbiHTMLBench>
- **入库日期：** 2026-11-09
- **核对范围：** 官方网站的项目说明、论文/代码与数据入口；配合代码主仓 README 验证开放范围。
- **通读：** 未完成

## 关键信息

- **1,115 张**页面图像，含 **13 类**注释对象；其中 **263 张**涉及表格，评价多区域页面而非纯文本抽取。
- **客观评分**对齐 HTML 解析后的文本块、表格、公式和布局关系；代码中的输出包括 `table_teds`、`table_teds_s`、`formula_cdm`、`text_edit_distance`、`layout_graph_similarity`。
- **主观评分**使用冻结 OLS5 多裁判代理协议，记录五名 judge 的成对偏好，再用软 Bradley–Terry 排序；论文另有人工标定研究。主观排名代表**偏好代理分**，不能视作客观精度或人类正确率。
- 同一模型可能在结构精度和外观偏好上排序不同，不应将两者合成未定义的单一“网页质量”。

## 开放与复现边界

README 提供评分实现和标注 JSON，但主观评测的原始 judge 结果、图像及人工注释需额外供应；主观 replay 本身不重新调用模型或生成新判断。

## 对 wiki 的映射

- [AmbiHTMLBench 主实体](../../wiki/entities/paper-ambihtmlbench.md)
- [论文归档](../papers/ambihtmlbench_arxiv_2611_06449.md)
- [代码仓归档](../repos/ambihtmlbench.md)
