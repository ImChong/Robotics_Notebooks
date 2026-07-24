# yuhao-zhou.com/vtap/（VTAP Gripper 项目页）

> 来源归档（ingest）

- **标题：** VTAP Gripper: Synergizing Fingertip Sensing and a Visuo-Tactile Active Palm for Dexterous In-Hand Manipulation
- **类型：** site / project-page
- **URL：** <https://yuhao-zhou.com/vtap/>（canonical；`https://yuhochau.github.io/vtap/` 301 至此）
- **论文：** [arXiv:2607.15448](https://arxiv.org/abs/2607.15448) — 归档见 [`sources/papers/vtap_gripper_arxiv_2607_15448.md`](../papers/vtap_gripper_arxiv_2607_15448.md)
- **代码：** **无** VTAP 官方仓（见下方开源核查）
- **入库日期：** 2026-07-24

## 一句话摘要

普渡大学（Purdue）× 哥伦比亚大学（Columbia）项目页：展示 **13-DoF 三指触觉反应夹爪**、**视触觉主动掌（VTAP）**、**手势条件遥操作重定向**，以及反应抓取 / 注射器手内操作 / 手内 singulation / 视触觉 peg-in-hole 等实验视频。

## 开源状态（步骤 2.5 · 2026-07-24）

| 核查项 | 结论 |
|--------|------|
| 项目页头部 / Resources | 可见 **Paper**（arXiv）；**无** 指向 VTAP CAD / 控制 / 遥操作栈的 GitHub |
| 「Code」按钮 | HTML 中仍残留指向 [RoboVerseOrg/ViTacFormer](https://github.com/RoboVerseOrg/ViTacFormer) 的模板链（页脚亦列 ViTacFormer / Nerfies 为站点模板）；**非** 本文实现仓 |
| 论文 PDF / HTML | 仅列项目页；未声明「code will be released」URL |
| 相关开源传感 | 指尖阵列引用 **FlexiTac**（[flexitac.github.io](https://flexitac.github.io/)，Huang & Li）；属传感栈上游，**不构成** VTAP 夹爪开源 |

**结论：确认未开源**（截至 2026-07-24）。读者勿把模板 Code 按钮当作可复现入口。

## 公开信息要点

- **机构 / 会议：** Purdue Edwardson IE + Columbia CS；**IROS 2026**；2026 ASME SMRDC Finalist。
- **硬件叙事：** Fin-Ray 顺应三指（每指 4 轴 Dynamixel）+ **50 mm** 行程主动掌；掌上 USB 相机通过 LED 开关在 **远场视觉 / 光学触觉** 间切换，无需机械换模。
- **遥操作：** Meta Quest 3 → 手势选 cage/power/pinch 子空间 → 指尖位置/朝向优化重定向；singulation 另映射拇指–食指滚动到内收/外展。
- **实验视频板块：** YCB/脆弱物反应抓取、注射器重定向与柱塞按压、绿豆/软糖/钢珠等手内 singulation、1 mm 公差 peg-in-hole。

## 为何值得保留

- **非 PDF 证据：** 设计爆炸图、掌模态切换与多任务视频比摘要更能支撑「指–掌协同 vs 高 DoF 拟人手」选型判断。
- **与论文互证：** 项目页与 arXiv Abstract / §IV 任务集合一致；canonical URL 已从旧 `yuhochau.github.io` 迁到作者站。

## 关联资料

- 论文归档：[`sources/papers/vtap_gripper_arxiv_2607_15448.md`](../papers/vtap_gripper_arxiv_2607_15448.md)
- Wiki：[`wiki/entities/paper-vtap-gripper.md`](../../wiki/entities/paper-vtap-gripper.md)
