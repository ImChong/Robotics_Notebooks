# What did we learn from 17K hours of open-source robot data?（CamX / CamUVA 原文核验记录）

- **类型：** 研究原文线索（匿名项目）
- **项目页：** https://camx-anonymous.github.io/
- **数据集：** https://camx-anonymous.github.io/camx-dataset/
- **策略评测：** https://camx-anonymous.github.io/camuva-rollouts/
- **核查日期：** 2026-11-09
- **核对范围：** 已核对官网标题、CamX/CamUVA 两入口、两个公开 GitHub README、CamX 站点目录/元数据以及 CamUVA 评测页面中任务设置；未取得论文全文、实验附录与模型权重，不将网页概括当论文内容。
- **通读：** 未完成

## 身份与版本

CamX 官网显示标题为 “What did we learn from 17K hours of open-source robot data?”；CamX 仓库自述“anonymized for review”。在已核对的官网 HTML 与公开 README 中没有可确认的论文 PDF / DOI / arXiv 原文链接。本记录**不填猜测的作者、机构、论文编号或会议**。

官网当前公开数据口径取自 [data/pending.json](https://github.com/camx-anonymous/camx-anonymous.github.io/blob/main/data/pending.json) 的 2026-10-30 metadata：65 来源、20 平台、2,900 datasets、1,794,319 episodes、17,102.6 小时；该文件另说明匿名论文附录保留非商业数据源，总数为 70 来源、24 平台组、4,469 datasets、2,450,988 episodes、25,850.7 小时。论文附录本身尚未获得，**第二组数字仅可归因于官方仓库的备注，不等于独立阅读全文核对**。

## 目前能证实的研究材料

- CamX 数据站点提供按机器人形态/来源浏览、任务字幕、相机投影演示和单 episode LeRobot v3 样本下载。
- CamUVA rollout 展示页在源码中列出同分布与跨本体评测任务；其成功标记来自视频索引，不是第三方复测。
- 不存在由已读站点源码足以支持的 CamUVA 架构细节、训练超参数、统计显著性或完整消融结论。

## 待完成

1. 获取论文公开 PDF 或正式 OpenReview/arXiv 版本，逐节阅读方法、训练、实验、附录并核对主要数值。
2. 核实 CamUVA 模型权重、训练代码与运行入口是否正式发布。
3. 若身份正式披露，核对相关 20K 小时标题是否同一研究的版本变化，**当前不合并**。

## 对 Wiki 的映射

- [CamX / CamUVA 主实体](../../wiki/entities/project-camx-camuva.md)
- [官方主页](../sites/camx-anonymous-github-io.md)
- [CamX 仓库](../repos/camx-dataset-site.md)
- [CamUVA 仓库](../repos/camuva-rollouts.md)
