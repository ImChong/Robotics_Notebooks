# Real-World Cooperative Bimanual Dexterous Grasp of Large Objects from Single-View Observations（arXiv:2608.10383）

> 来源归档（ingest）

- **标题：** Real-World Cooperative Bimanual Dexterous Grasp of Large Objects from Single-View Observations
- **类型：** paper / bimanual / dexterous-grasp / ddpm / force
- **arXiv：** <https://arxiv.org/abs/2608.10383>
- **会议：** IROS 2026
- **代码：** <https://github.com/zhangdana483/real_bi_dex_grasp>（归档见 [`sources/repos/real-bi-dex-grasp.md`](../repos/real-bi-dex-grasp.md)）
- **作者：** Ziming Li、Mingxuan Wu、Jiaqi Zhang、Hongfei Li、Yan Gan、Deqiang Ouyang、Ning Wang
- **入库日期：** 2026-08-18
- **一句话说明：** 单视角点云生成双臂协作灵巧抓取：多模态数据集 + DDPM 关节配置 + 运动规划与在线力细化，不依赖完整 3D 模型。

## 开源状态（步骤 2.5）

- **无独立项目页**；以 GitHub 为准。
- **仓库核查（2026-08-18）：** [zhangdana483/real_bi_dex_grasp](https://github.com/zhangdana483/real_bi_dex_grasp)（代码 Apache-2.0）含 `ddpm_model/train_ddpm.py`、`infer_ddpm.py`、遥操作采集、数据样例；全集约 40GB 百度网盘。
- **结论：** **已开源、可辨识训练/推理入口**；完整数据集不在 GitHub。

## 摘录

既有双臂工作多在仿真或顺序操作。本文：分割点云 → DDPM 出关节级抓取配置 → 规划执行并用力信号在线细化。摘要称未见物体跨几何/位姿成功率高，消融确认各模块贡献（未给百分比表）。

**对 wiki 的映射：** [`wiki/entities/paper-real-bi-dex-grasp.md`](../../wiki/entities/paper-real-bi-dex-grasp.md)；交叉 [双臂操作](../../wiki/tasks/bimanual-manipulation.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（训练脚本可辨识）
