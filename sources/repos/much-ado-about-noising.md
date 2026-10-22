# simchowitzlabpublic/much-ado-about-noising

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/simchowitzlabpublic/much-ado-about-noising>
- **对应论文：** <https://arxiv.org/abs/2512.01809>
- **项目页：** <https://simchowitzlabpublic.github.io/much-ado-about-noising-project/>
- **许可：** MIT
- **核查日期：** 2026-10-22
- **一句话说明：** 基于 PyTorch 的行为克隆实现，覆盖 flow matching 及相关生成式策略与 MIP。

## 仓库内容

README 提供 Robomimic、Kitchen 与 PushT 任务入口，包含 flow matching、回归、MIP、TSD、CTM、PSD、LSD、ESD、Mean Flow 等训练目标，并链接文档、数据集与 checkpoints。训练示例使用 `examples/train_robomimic.py`；评估示例加载公开 checkpoint。

## 复现边界

仓库说明依赖升级与代码整理后，复现实验数值可能与论文不同；论文原始结果使用 cleandiffuser Robomimic 环境与数据。README 展示仿真训练与评估示例，本归档不推断其包含真机部署栈。

## 关联资料

- [官方项目页归档](../sites/much-ado-about-noising-project.md)
- [论文摘录](../papers/much_ado_about_noising_arxiv_2512_01809.md)
