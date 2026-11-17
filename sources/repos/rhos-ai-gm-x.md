# rhos-ai/GM-X：The Great March 官方仓库

- **仓库：** https://github.com/rhos-ai/GM-X
- **论文：** https://arxiv.org/abs/2601.11421
- **项目页：** https://www.rhos.ai/research/gm-100
- **数据集：** [Cobot Magic / LeRobot](https://huggingface.co/datasets/rhos-ai/gm100-cobotmagic-lerobot) · [Dobot Xtrainer / HDF5](https://huggingface.co/datasets/rhos-ai/gm100-xtrainer)
- **核查日期：** 2026-11-17
- **通读：** 已完成
- **核对范围：** GitHub 默认分支 README 全文与根目录清单（README.md、assets/、.gitignore）；核对论文、任务设计说明、数据下载表与开源入口。
- **定位：** GM-X 计划的官方材料与数据导航；GM-100 是该计划第一阶段的 100 项细粒度机器人操作评测。

## 实际可用资源与限制

- README 描述 HAKE + affordance 系统化长尾任务设计与 >13,000 条专家轨迹。
- Agilex Cobot Magic 提供 LeRobot 下载；Dobot Xtrainer 提供 HDF5 下载，LeRobot 一栏标记 Processing。
- **没有发现可执行的 GM-100 训练、推理、评测入口**；仓库根目录没有 scripts、训练包或环境说明，不能把官方论文中「使用 LeRobot/OpenPi 微调」误解成 GM-X 仓已发布复现代码。
- README 虽显示 Apache-2.0 徽章，但根目录未列 LICENSE 文件；具体资产许可仍需以各数据集卡及后续正式授权文本为准，不能据此一概断言全部素材均受 Apache-2.0 授权。

## 对知识库映射

- [论文通读归档](../papers/gm100_arxiv_2601_11421.md)
- [官网核查](../sites/gm100-rhos-ai.md)
- [独立论文实体](../../wiki/entities/paper-gm-100.md)
