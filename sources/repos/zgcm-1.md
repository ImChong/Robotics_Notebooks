# zgcagi/ZGCM-1

> 来源归档（代码仓库；草稿）

- **代码：** <https://github.com/zgcagi/ZGCM-1>
- **报告：** <https://arxiv.org/abs/2609.13356v2>
- **权重：** <https://huggingface.co/zgcagi/ZGCM-1-7B>
- **数据：** <https://huggingface.co/datasets/zgcagi/ZGCM-1-Data>
- **许可证：** 仓库 MIT；数据集需另行核对其访问条件及上游许可。
- **入库日期：** 2026-11-03

## 已核对入口

仓库提供 `data-process/`、`pretrain/`、`midtrain/`、`sft/`、`rl/` 及各自运行文档。预训练 README 指向 `pretrain/scripts/launch_train.sh` 和分阶段配置；该脚本读取配置、校验数据与运行环境，再通过 torchrun 启动配置指定的训练程序。SFT README 提供配置验证和多 GPU 运行入口。

数据集文件需接受 Hugging Face 访问条件；实际数据组成及上游许可须在取得访问后进一步核查。

## 待办

继续阅读训练入口调用链、数据集构造及优化器实现；通读报告后在统一项目实体页补流程图和源码运行时序图。

## 关联资料

- [报告归档](../papers/zgcm_1_arxiv_2609_13356.md)
