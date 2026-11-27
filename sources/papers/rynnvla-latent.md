# RynnVLA-Latent: Scaling Robotic Manipulation Foundation Models with 3D-Grounded Latent Actions

> 论文来源归档；结果以指定版本和实验表为准。

- **论文：** [arXiv v1](https://arxiv.org/abs/2610.34356v1) · [PDF](https://arxiv.org/pdf/2610.34356v1) · [HTML](https://arxiv.org/html/2610.34356v1)
- **作者：** Haoyu Zhao、Xingyue Zhao、Zhian Su、Bohan Hou、Yinzhou Tang 等。
- **机构：** 阿里巴巴达摩院、香港具身智能实验室、香港中文大学、湖畔实验室、阿里巴巴集团、北京协和医学院（论文署名）。
- **版本：** arXiv 记录 v1 提交于 2026-10-25 UTC；PDF 首页正文另外标注 10-27。已有 v2，本条不混用 v2 数据。
- **项目页：** <https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io/>
- **代码：** <https://github.com/alibaba-damo-academy/RynnVLA-Latent>
- **入库日期：** 2026-10-28；**复核日期：** 2026-11-27。
- **通读：** 已完成
- **核对范围：** v1 正文 §§1–5、表 1/3–11、附录 A–F（评测设置、真机采集、规模对照及潜动作消融）；与官方项目页、模型卡和固定提交源码交叉核对。未复跑论文大规模训练或机器人评测。

## 实验定位

表 1 汇总 25 个来源、30,117.68 小时视频；论文报告离线生成 745.52M 潜动作描述符。LARY 表 3 的 0.38 是冻结表示的相对动作回归平均 MSE，不是米制运动误差。4B 的 98.8% LIBERO、84.7% LIBERO-Plus、51.9% VLABench、88.6% RoboTwin 2.0 分别看表 4–7。RoboTwin 摘要和表 6 为 88.6%，§1 与 §5 另写 88.3%；本页采用可由两列结果复核的表 6（Clean 88.7%、Randomized 88.4%）。真机表 8 为两种具身完成各自的后训练后测六项任务，每项 35 次。

## 对 wiki 的映射

- [RynnVLA-Latent 实体页](../../wiki/entities/paper-rynnvla-latent.md)
- [资料和模型合集](../sites/rynnvla-latent.md)
