# HumanCOIN 官方项目与数据发布入口（2026-11-23 核验）

- **类型：** 官方项目与发布入口来源归档
- **项目页：** https://flagopen.github.io/HumanCOIN/
- **官方文档：** https://flagopen.github.io/HumanCOIN-Doc/
- **官方数据门户：** https://flagopen.github.io/HumanCOIN-DataManager/
- **官方 GitHub：** https://github.com/FlagOpen/HumanCOIN
- **论文：** https://arxiv.org/abs/2611.26434v1
- **数据组织：** https://huggingface.co/HumanCOIN
- **ModelScope：** https://www.modelscope.cn/organization/HumanCOIN
- **核查日期：** 2026-11-23
- **核对范围：** 已查官方 GitHub README.md/README_zh-CN.md 中的项目/文档/数据门户链接、授权协议、示例数据仓库/下载流程与开放限制；项目首页在本次公开读取接口中未能直接获取正文，**未声称通读项目首页或独立核实门户完整清单**。
- **通读：** 未完成

## 发布状态与访问边界

- 官方 README 报告 12,500 h、三层注视–EMG–EEG 数据、约 1.35B 个 30 fps 主帧、6 个采集地点；论文原文对这些统计给出方法与局限。
- **代码可用**：Python `humancoin` 包、安装与 API 文档、下载 CLI、读取/写入示例与可视化器存在于官方仓库。
- **数据发布为多仓库、部分需访问许可**：README 提供 65 MB 的 L1 示例和 `humancoin-download` 命令，说明大多数数据仓库 gated。不得把“提供组织页面/门户”写成“全量匿名开放直接下载”。
- **许可分离**：数据 CC BY-NC 4.0（非商业），代码 Apache-2.0。
- **核验盲区**：当前工具无法打开项目首页 HTML；上述项目发布信息核对自官方 GitHub README 和 arXiv 原文，不对页面动态内容做额外保证。

## Wiki 映射

- [HumanCOIN 论文实体](../../wiki/entities/paper-humancoin-human-intent-egocentric-manipulation.md)
- [论文来源归档](../papers/humancoin_human_intent_egocentric_manipulation_arxiv_2611_26434.md)
- [官方工具代码归档](../repos/humancoin-dataset-sdk.md)
