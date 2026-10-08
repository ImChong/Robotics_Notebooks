# Video Prediction Policy 官方项目页

- **类型：** site
- **项目页：** https://video-prediction-policy.github.io/
- **论文：** https://arxiv.org/abs/2412.14803
- **代码：** https://github.com/roboterax/video-prediction-policy
- **核查日期：** 2026-10-08
- **机构：** 清华大学、加州大学伯克利分校、上海人工智能实验室、上海期智研究院、星动纪元

## 页面核查

项目页列有 arXiv 与 Code 链接，标注 ICML 2025 Spotlight。方法为两阶段：先用多来源操作视频适配文本条件视频预测，再聚合预测视觉表征学习动作模型；不是只用静态图像编码器。

项目页的 CALVIN 相对提升写为 41.5%，当前 arXiv v2 摘要为 18.6%；二者口径存在差异，不能合并为同一个已核验数字。本文以官方 README 的 ABC-D 平均完成链长 4.33 作为明确的复现目标，并分别保留来源口径。

## 资源开放范围

[仓库归档](../repos/video-prediction-policy.md)记录当前训练/评测入口与公开模型链接。示例和 latent 特征不等于全部真机原始训练数据；真机评测还依赖 XBot/XHAND 本体接口。

## 对 wiki 的映射

- [VPP 主实体](../../wiki/entities/paper-shenlan-wm-02-vpp.md)
