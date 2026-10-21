# ForeTac 项目页（foretac.github.io）

> 来源归档

- **标题：** ForeTac: Predictive Contact Guidance for Generative Robot Policies
- **类型：** site（项目页）
- **URL：** <https://foretac.github.io/>
- **论文：** <https://arxiv.org/abs/2610.18118> — 归档见 [`sources/papers/foretac_arxiv_2610_18118.md`](../papers/foretac_arxiv_2610_18118.md)（页面未链）
- **代码：** 页面标注 "Coming soon"，尚无 GitHub / Hugging Face 链接（2026-10-17 核查）
- **入库日期：** 2026-10-17
- **一句话说明：** 展示四任务平均结果、消融、延迟与引导诊断；作者匿名。

## 页面要点（2026-10-17 核查）

| 区块 | 内容 |
|------|------|
| 链接 | Paper、Code 均 "Coming soon"；视频 "Video unavailable"；BibTeX 作者为 "Anonymous" |
| 主结果（四任务各 20 次） | DP+ForeTac SR 71.3% / CSR 66.3%，DP 36.3% / 20.0%，RDP 58.8% / 48.8%；π0.5+ForeTac 67.5% / 62.5%，π0.5 46.3% / 36.3% |
| 消融（SR/CSR） | 冻结 ResNet-18 50.0 / 38.8；仅辅助损失 55.0 / 48.8；预测重排序 58.8 / 52.5；当前接触控制 60.0 / 51.3 |
| 延迟 | 每次重规划 132.46 ms（DP）对 261.26 ms（ForeTac）；基座 338.0M 参数，ForeTac 增 34.9M |
| 其他 | 预测触觉偏好序准确率 92.9%（记录触觉 91.2%）；引导诊断 293/305（96.1%）配对更新改善接触质量 margin |
