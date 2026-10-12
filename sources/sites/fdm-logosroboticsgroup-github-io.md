# FDM 项目页核查

- **类型：** 论文项目页
- **标题：** Forward Dynamics Model
- **链接：** <https://logosroboticsgroup.github.io/FDM/>
- **核查日期：** 2026-10-12
- **论文：** <https://arxiv.org/abs/2610.13117>（归档：[fdm_arxiv_2610_13117.md](../papers/fdm_arxiv_2610_13117.md)）
- **代码：** https://github.com/LogosRoboticsGroup/FDM（归档：[fdm.md](../repos/fdm.md)）

## 页面内容

- 一句话：从未来状态预测中学动作，推理时跳过未来帧生成。
- 页面称“代码、训练配置和评测说明在仓库中”，链接两个实现文件：`starVLA/model/framework/VLM4A/Pi05_causal.py`（FDM w/ VLM）与 `starVLA/model/framework/WM4A/WanMoT_causal.py`（FDM w/ VGM）。
- 页面含演示视频、架构对比与真机结果图；页面本身未提供 arXiv 链接，也没有 HuggingFace 链接。

## 开放结论

已开源（代码）：仓库存在、MIT 许可，含训练、评测与部署入口；数据集和训练好的 FDM checkpoint 不随仓库发布（README 自述，“public FDM checkpoint download links are not yet provided”）。

## 对 wiki 的映射

- [FDM 详情节点](../../wiki/entities/paper-fdm-forward-dynamics-models.md)
