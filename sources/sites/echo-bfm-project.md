# ECHO 官方项目页

- **类型：** site / official paper project
- **网址：** https://mwondering.github.io/echo-bfm/
- **论文：** https://arxiv.org/abs/2611.25778（2026-11-19 v1）
- **GitHub：** https://github.com/mwondering/echo-bfm（展示网站源码）
- **模型代码：** 未发现 ECHO 策略训练/推理/部署代码（截至 2026-11-23）
- **模型/数据：** 项目页未提供 ECHO checkpoint 或自有数据归档下载入口（截至 2026-11-23）
- **通读：** 已完成
- **核对范围：** 阅读 Overview、Method Pipeline、Results、Context Representation、Demo Gallery 与 GitHub 资源入口；查阅链接仓库 README 与根目录。
- **入库日期：** 2026-11-23

## 内容与开放边界

- MIAS 混合困难样本回退与广覆盖采样，训练 ECHO-PRE。
- CRA 从状态—动作历史用多步动态预测和物理邻域 soft alignment 编码环境；冻结先验后训练残差 actor。
- 官网表 2 的 BONES-SEED 高动态 SR：无 MIAS 0.4500、ECHO-PRE 0.7200、ECHO-N 0.7633。
- 演示库有 6 段特技、3 段地面交互、4 段地形交互、6 段 in-context 视频，共 19 段。
- GitHub 仓库 `scripts/build.py` 是**网页生成入口**，并不是论文中的 mjlab PPO 或 CRA 模型训练入口，不能据此作算法运行时序图。

## 对 wiki 的映射

- [ECHO 论文节点](../../wiki/entities/paper-echo-bfm.md)
- [论文全文核对](../papers/echo_bfm_arxiv_2611_25778.md)
- [展示网站仓库核查](../repos/echo-bfm-website.md)
