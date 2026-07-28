# Motion Turing Test / HHMotion 官方项目页

> 来源归档（项目页核查）

- **标题：** Towards Motion Turing Test: Evaluating Human-Likeness in Humanoid Robots
- **类型：** site / project-page
- **官方入口：** <http://www.lidarhumanmotion.net/mtt/>
- **论文：** <https://arxiv.org/abs/2603.06181>
- **机构：** 厦门大学、OPPO 研究院、上海科技大学
- **入库日期：** 2026-07-28
- **一句话说明：** 汇总 HHMotion 数据构建、人工评分协议、PTR-Net 基线与 CVPR 2026 结果。
- **数据集：** 页面介绍 HHMotion，但未列可点击下载地址。
- **代码：** 页面介绍 PTR-Net，但未列 GitHub 或代码包地址。
- **开源状态（2026-07-28 核查）：** **宣称将公开 / 尚未落地**；摘要写 dataset、code、benchmark “will be publicly released”，页面当前未给资源链接。

## 页面公开信息

- 1,000 个 5 秒片段，涵盖 15 类动作、11 种人形机器人与 10 名人类受试者。
- 统一转为 SMPL-X，30 名标注者按 0–5 分评价类人度；IAC 核查后保留 25 名一致标注者。
- PTR-Net 使用双向 LSTM、ST-GCN 与注意力池化预测类人度分数。

## 对 wiki 的映射

- 论文归档：[humanoid_pnb_towards-motion-turing-test.md](../papers/humanoid_pnb_towards-motion-turing-test.md)
- 实体页：[paper-notebook-towards-motion-turing-test.md](../../wiki/entities/paper-notebook-towards-motion-turing-test.md)
