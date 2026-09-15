# artmanip.github.io（ArtManip 项目页）

- **标题：** ArtManip: Category-Level Articulated In-Hand Manipulation — 官方项目页
- **类型：** site / project-page
- **URL：** <https://artmanip.github.io/>
- **入库日期：** 2026-09-15
- **配套论文：** [ArtManip（arXiv:2609.12498）](https://arxiv.org/abs/2609.12498) — 归档见 [`sources/papers/artmanip_arxiv_2609_12498.md`](../papers/artmanip_arxiv_2609_12498.md)
- **配套代码：** <https://github.com/youngcv/artgym>（仓库名 **ArtGym**）

## 一句话摘要

BIGAI / 浙大 / 清华 / 北大联合工作 **ArtManip** 官方站点：类别级 **铰接物体手内操作**——在 **自由漂浮基座** 上既要 **抓稳** 又要 **驱动内部关节**（开合、滑动等）；程序化两连杆铰接资产 + 功能抓取合成 → Teacher（特权状态 + 课程 + 关节物理随机化）→ Student（历史潜表示蒸馏）→ **四类物体仿真泛化 + 12 个真机物体零样本 sim2real**。

## 公开信息要点（截至 2026-09-15）

- **作者与机构：** Yang Yang、Tengyu Liu、Puhao Li、Zeyuan Chen、Yuyang Li、Xingwan Wang、Yingying Wu、Zhaopeng Cui、Siyuan Huang；**浙江大学 CAD&CG 国家重点实验室**、**北京通用人工智能研究院（BIGAI）**、**清华大学**、**北京大学**。
- **arXiv：** [2609.12498](https://arxiv.org/abs/2609.12498)（2026-09-11）。
- **核心问题（摘要原文）：**
  1. 控制铰接物体 **内部自由度** 与在 **自由漂浮基座上维持抓取稳定** 强耦合；
  2. 大规模获取多样物体模型与 **任务导向功能抓取** 成本高，而系统对 **初始构型** 敏感。
- **方法管线（页面四步）：**
  1. **Primitive articulated assets** — 用 box 原语程序化生成 **单 prismatic 或 revolute 关节** 的两连杆物体；
  2. **Functional grasp synthesis** — 类别级接触区域模板合成多样 **任务导向初始抓取**；
  3. **Teacher policy learning** — 特权状态、**奖励课程**、**铰接物理随机化**；
  4. **Student deployment** — 从可观测历史 **蒸馏潜表示**，零样本真机执行。
- **实验（页面口径）：** 四类物体类别仿真泛化；**12 个真实物体**（多样形状与关节机构）零样本迁移；演示含 knife / stapler / tong 等。
- **开源（步骤 2.5 复核，2026-09-15）：** 项目页 **Code** 按钮链至 <https://github.com/youngcv/artgym>；README 标明为 ArtManip 官方实现，含 Isaac Gym 环境、抓取验证、Teacher 训练、Student 蒸馏与评测脚本。**已开源**（训练/仿真管线；真机部署接口以仓库为准）。

## 为何值得保留

- **一手入口**：论文 PDF、项目视频、代码仓库均从该页链出；适合 curator 核验「抓稳 + 推动」双目标与开源边界。
- **任务定义清晰**：区别于纯手内重定向（改物体 6D 位姿），ArtManip 强调 **铰接内部 DoF** 在 **无桌面支撑** 条件下的类别级泛化。
- **可复现管线**：ArtGym 给出从 `make_data` / `func_lygra` 生成资产与初始抓取 → 验证 → Teacher → 蒸馏 Student 的完整脚本链。

## 关联资料

- 论文归档：[`sources/papers/artmanip_arxiv_2609_12498.md`](../papers/artmanip_arxiv_2609_12498.md)
- 代码归档：[`sources/repos/artgym.md`](../repos/artgym.md)

## 对 wiki 的映射

- [ArtManip（论文实体）](../../wiki/entities/paper-artmanip.md)
