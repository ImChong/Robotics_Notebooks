# SimpleTouch（arXiv:2610.02784）

- **标题：** SimpleTouch: Can Vision-Language-Action Models Master Contact-Rich Manipulation Without Tactile Policy Pretraining?
- **作者：** Chen Yang、Linzhe Shi 等
- **机构：** 清华大学、阿里巴巴集团高德、香港理工大学
- **论文：** <https://arxiv.org/abs/2610.02784>
- **项目页：** <https://simpletouch-robot.github.io/>；[项目归档](../sites/simpletouch.md)
- **代码：** <https://github.com/ccyydd/SimpleTouch>；[仓库归档](../repos/simpletouch.md)
- **权重：** <https://huggingface.co/cyang203912/SimpleTouch>
- **发布日期：** 2026-10-02
- **入库日期：** 2026-10-27

## 摘要要点

- 在预训练 π₀.₅ 上加入触觉专家，使用冻结的预训练触觉编码器全部 token。
- 仅用任务示范，以动作监督和多时域未来触觉潜变量预测联合训练；不另做触觉策略预训练或独立跨模态对齐。
- UniVTAC 六项任务每项 50 条示范，平均成功率 77.5%；FTP-π₀.₅ 为 45.2%，FTP-1 为 66.7%。四项真机任务平均 71.3%，较 FTP-1 高 8.8 个百分点。

## 对 wiki 的映射

- [SimpleTouch](../../wiki/entities/paper-simpletouch.md)：触觉专家、实验边界及复现入口。
