# CamX / CamUVA 匿名官方主页

- **入口：** https://camx-anonymous.github.io/
- **类型：** 研究项目页（匿名审稿）
- **核查日期：** 2026-11-09
- **核对范围：** 首页 HTML、CamX 数据集子页源码、CamUVA Rollouts 子站源码；未获取研究论文全文。
- **通读：** 未完成
- **代码（CamX 网站/数据工具）：** https://github.com/camx-anonymous/camx-anonymous.github.io
- **代码（CamUVA 展示/视频处理）：** https://github.com/camx-anonymous/camuva-rollouts
- **数据：** GitHub Releases 的单 episode 示例归档；完整数据页面注明 “Full dataset upon official release”。

## 页面结构

官网把“Click to Browse CamX Dataset”和“Click to Watch CamUVA Rollouts”分为两条入口。CamX 子页 metadata 的描述为跨手持、单臂、双臂和移动操作平台汇集的 camera-aware LeRobot-v3 布局；CamUVA 子站展示策略在不同本体/任务下的视频。

## 数据许可与规模（不要合并两个版本）

[CamX data/pending.json](https://github.com/camx-anonymous/camx-anonymous.github.io/blob/main/data/pending.json) 的 2026-10-30 汇总：65 sources、20 platforms、2,900 datasets、1,794,319 episodes、17,102.6 h。其说明指出裁剪前的论文附录总计 70 sources、24 groups、4,469 datasets、2,450,988 episodes、25,850.7 h；网站保留可商业使用许可来源（RH20T 仅 RH20T-C 场景）。

网页提供来源引用及 license 确认，单 episode 的 LeRobot v3 样本可按所选数据集生成下载命令；不能据此推断全量训练集已公开或可自由商用全部原始来源数据。

## 对 wiki 的映射

- [CamX / CamUVA 主实体](../../wiki/entities/project-camx-camuva.md)
- [CamX 数据站点实现](../repos/camx-dataset-site.md)
- [CamUVA rollout 站点实现](../repos/camuva-rollouts.md)
- [论文身份核验](../papers/camx-17k-open-source-robot-data.md)
