# Articraft: An Agentic System for Scalable Articulated 3D Asset Generation

- **作者：** Matt Zhou, Ruining Li, Xiaoyang Lyu, Zhaomou Song, Zhening Huang, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi, Shangzhe Wu
- **机构：** Nanyang Technological University (NTU)、University of Oxford、University of Cambridge
- **版本 / 载体：** arXiv:2605.15187v1 [cs.CV]（2026-05-14 提交）
- **原文：** https://arxiv.org/html/2605.15187 （摘要页：https://arxiv.org/abs/2605.15187 ）
- **项目页 / 代码：**
  - 项目主页：https://articraft3d.github.io/ （页面提示开发已迁移至新仓库）
  - 官方源码：https://github.com/articraftresearch/Articraft （**已开源**，Apache-2.0 协议，后续替代早期原型 `mattzh72/articraft`）
- **核对范围：** 通读 `paper_text.txt` 全文（1527 行，涵盖 Abstract、Sections 1–7 全部正文、Table 1–3、Appendix A–E 含 Table 4–9 及 Figure 1–20 题注），并核对 `project_page.txt` 与官方仓库快照 `context.json`（含 `README.md` 与 `src/articraft/` 目录树）。
- **通读：** 已完成

## 原文核对笔记

- **核心范式转换**：将可动关节（articulated）3D 资产生成归约为编写单文件 Python 构建程序 `model.py`。模型通过调用面向领域设计的 Articraft SDK（`ArticulatedObject`、`Part`、`ArticulationType`、`TestContext` 等）与受限执行外壳（Harness）进行「编辑—编译校验—几何探针—修复」闭环，无需启动 Blender 或执行高成本的多视角渲染视觉反馈。
- **关键量化数据**：
  - **Articraft-10K 数据集**：覆盖 245 个细分类别（15 个超类）；经人工 1–5 分制三项准则评分（几何真实感、预期关节完备性、物理运动合理性），保留 $\ge 4$ 分资产，共从 10,909 次生成中筛选保留 10,018 个（保留率 91.8%；GPT-5.4 89.4%、GPT-5.5 95.5%、Gemini 3.1 Pro 96.3%）。
  - **API 成本与本地算力**：生成流水线纯 CPU 执行，无需 GPU。10,880 条成本记录显示总 API 成本 $12,386.99，单次尝试均值 $1.139（中位 $1.030），平均 16.8 轮（中位 15 轮），Prompt Cache 命中率 85.7%。
  - **用户偏好实验**：在 PartNet-Mobility 46 类 × 5 prompt（230 题）上由 125 名参与者完成 5,000 次盲测排序，Articraft（GPT-5.5 / GPT-5.4）击败 Articulate-Anything、PhysX-Anything、URDF-Anything+ 及直接使用通用编程智能体 Codex（GPT-5.5 单独使用排名倒数第二）。
  - **下游模型增强（Table 3 / Table 6）**：用 Articraft-10K 增广训练 Particulate 后，在 Lightwheel 14 类基准上，静止姿态分割 gIoU 由 0.332 升至 0.394、mIoU 由 0.576 升至 0.607，关节几何 gIoU 由 0.305 升至 0.361，且在原本训练集中缺失的域外类别上增益最明显。
- **对 wiki 的映射：** [Articraft 实体页](../../wiki/entities/paper-agentic-2605-15187.md)
