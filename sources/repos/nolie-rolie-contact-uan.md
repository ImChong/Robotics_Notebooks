# nolie-rolie/contact-uan

- **URL：** <https://github.com/nolie-rolie/contact-uan>
- **项目页：** <https://contact-uan.csail.mit.edu/>
- **论文：** [Contact-UAN（arXiv:2610.18991）](../papers/contact_uan_arxiv_2610_18991.md)
- **主机体：** Unitree G1

## 一句话说明

Contact-UAN 官方实现：从自采 G1 rollout 训练接触一致状态估计与执行器残差，再在冻结 UAN 的仿真中训练任务策略并导出控制器。

## 仓库入口与复现边界

- [README](https://github.com/nolie-rolie/contact-uan) 指向硬件流程、数据格式和部署指南。
- 硬件管线包含 WTW 训练、G1 数据采集、六类本体感知估计器训练、轨迹检查/裁剪、重标、UAN 与下游策略训练。
- 部署参考涵盖策略 TorchScript 导出、配置校验与 G1 控制接口；UAN 不随策略部署。
- 预训练策略与硬件数据集未包含，端到端复现需要真实 G1 和用户自采数据。

## 对 wiki 的映射

- [Contact-UAN 实体页](../../wiki/entities/paper-contact-uan.md)
