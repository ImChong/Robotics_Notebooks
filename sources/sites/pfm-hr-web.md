# gaoyukang33.github.io/PFM-HR.web（PFM-HR 项目页）

- **标题：** PFM-HR: Pose Flow Matching for Humanoid Robots
- **类型：** site / project-page
- **URL：** <https://gaoyukang33.github.io/PFM-HR.web/>
- **配套论文：** [PFM-HR（arXiv:2608.03227）](https://arxiv.org/abs/2608.03227) — 归档见 [`sources/papers/pfm_hr_arxiv_2608_03227.md`](../papers/pfm_hr_arxiv_2608_03227.md)
- **代码：** <https://github.com/gaoyukang33/PFM-HR> — 归档见 [`sources/repos/pfm-hr.md`](../repos/pfm-hr.md)（截至 2026-08-08 为 Coming Soon 占位）
- **机构展示：** 1 HKUST(GZ) · 2 Noitom Robotics · 3 SIGS, Tsinghua University · 4 Google
- **入库日期：** 2026-08-08

## 一句话摘要

PFM-HR 官方项目站：用无序姿态预训练的 **Flow Matching 姿态先验** + **Pose Geometry Score（PGS）** 调制跟踪奖励；演示单轨迹高动态技能、通用运动跟踪，以及挂到 BeyondMimic 后的真机 Kick Combo / Spin Kick。

## 公开信息要点（截至入库日）

- **头部按钮：** Paper（PDF）、arXiv、Code（GitHub）。
- **Abstract：** 对比时序先验（需有序 clip）与姿态先验（弱于姿态转移）；提出冻结可复用 PFM-HR + PGS。
- **Single Motion Tracking：** Backflip（相对 ADD w/ PDF-HR **−14.3%** 样本、**−6.3%** 位置误差；ADD 不收敛）、Double Kong（**−28.8%** 样本、**−9.7%** 位置误差）。
- **General Motion Tracking：** 跨 episode 长度成功率 / 样本效率曲线。
- **Real World：** BeyondMimic 管线仅仿真训练加冻结先验；相对原 BeyondMimic，达目标成功率样本 Spinkick **−24.2%**、Kick combo **−15.1%**。
- **BibTeX：** `@misc{gao2026pfmhrposeflowmatching, eprint={2608.03227}, ...}`。

## 开源核查

- 项目页 **有** Code 链接；仓库 tip 仅 MIT LICENSE + README「Coming Soon」。
- **结论：** 入口已挂、实现待发布；勿写已可复现。

## 为何值得保留

- 与 PDF-HR / SMP / BeyondMimic 形成「冻结姿态几何先验」选型锚点。
- 视频与定量百分比是论文 Table 之外的读者入口。

## 关联资料

- 论文归档：[`sources/papers/pfm_hr_arxiv_2608_03227.md`](../papers/pfm_hr_arxiv_2608_03227.md)
- 代码仓库：[`sources/repos/pfm-hr.md`](../repos/pfm-hr.md)
- wiki：[`wiki/entities/paper-pfm-hr.md`](../../wiki/entities/paper-pfm-hr.md)
