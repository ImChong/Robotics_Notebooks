# GitHub 仓库：QwenLM/Qwen-RobotManip

- **仓库：** https://github.com/QwenLM/Qwen-RobotManip
- **关联项目：** Qwen-RobotManip
- **开源状态：** 未开源可运行代码与权重（仅发布项目介绍 README 与演示多媒体资源；仓库明确声明暂无发布模型权重计划）
- **核对范围：** 已通读官方仓库 `Qwen-RobotManip` 完整目录结构（仅含 `README.md` 与 `assets/`）及 `README.md` 1–167 行全部文本（含 BibTeX 引用与发布状态声明）
- **通读：** 已完成

## 仓库核查结论

1. **目录与文件构成**
   - 当前远程仓库仅包含文档说明 `README.md` 与 `assets/` 静态图片/视频资源，不含模型权重、训练脚本、评测脚本或推理服务代码。
   - **官方发布声明原文要点**：`Note: There is currently no plan to release the model weights for Qwen-RobotManip or Qwen-RobotNav.`。

2. **核对出的正式 BibTeX 文献信息**
   - `README.md` 提供的规范引文为：`Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models`，`arXiv:2606.17846`（2026 年），作者署名 `Qwen Team`。
   - 数据、工程管线与架构分析只能依据技术报告正文还原，无法通过源码静态分析验证底层张量并行或 RTC 调度代码细节；因此在 wiki 实体页 `源码运行时序图` 区块须如实标注“不适用”。
