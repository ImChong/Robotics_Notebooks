# DiffWBC / OneSteer 项目主页与源码开放核查

- **项目主页：** https://diffwbc.github.io/
- **主论文：** https://arxiv.org/abs/2610.41354（[论文全文通读归档](../papers/diffwbc_arxiv_2610_41354.md)）
- **入库核查日期：** 2026-11-10
- **类型：** site / project-page / humanoid motion tracking
- **核对范围：** 通读项目页方法、真机视频说明、引导任务、常规跟踪与自评结果、开源按钮和页脚 BibTeX；核对 arXiv v1 标题与主要数字。
- **通读：** 已完成

## 开放状态

页面头部仍显示 **OneSteer**，署名 Anonymous、Paper (coming soon)、**Code (after review)**；页脚为匿名 OneSteer 2027 占位 BibTeX。页面没有提供可验证的 DiffWBC 官方 GitHub 源码、训练/推理入口、权重或数据下载链接（截至 2026-11-10）。

**结论：待发布 / 目前未见可运行源码**；不可将论文中“导出 ONNX 真机部署”误写为已发布 ONNX 权重。本站因此**不建 sources/repos/ 占位文件**，wiki 的源码时序图标记“不适用”。

## 页面内容与论文口径差异

| 项目主页 | arXiv:2610.41354v1 | 处理 |
|---|---|---|
| 匿名 OneSteer 名称 | DiffWBC、实名作者、2026-10-30 | 使用论文标题和作者；注明主页尚未更新 |
| BONES 从头训练 local error 22.6 mm | Table 2：20.1 mm | 指标使用论文 v1 |
| 展示 50 Hz、G1 真机跟踪 | §4.2：9 种动作 × 3 次无摔倒 | 写清真机仅验证常规跟踪 |
| 展示顶棚、栏杆、弱电机、托盘等 | §4.3–4.4 中仿真多目标引导 | 不写作已在真机验证 |

## 对 wiki 的映射

- [DiffWBC 统一论文实体](../../wiki/entities/paper-diffwbc.md)
- [论文通读归档](../papers/diffwbc_arxiv_2610_41354.md)
