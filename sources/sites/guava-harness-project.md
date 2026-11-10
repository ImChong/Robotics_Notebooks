# Guava — 官方项目主页核验

- **项目：** Guava: Distilling Frontier VLMs into a Compact Agent through a Robotic Manipulation Harness
- **主页：** <https://guava-harness.github.io/>
- **论文：** <https://arxiv.org/abs/2606.18363>（本文以 v3 为准）
- **代码：** <https://github.com/hdacnw/guava-release>
- **模型权重：** <https://huggingface.co/AIcell/guava-v13b-qwen3.5-4b>
- **训练数据：** <https://huggingface.co/datasets/AIcell/Guava-4B-SFT-data>
- **核对范围：** 项目页系统概览、效果演示、论文/Code/Model 等链接与 Results；将数值与论文 v3、发布仓库 README 交叉对照。
- **通读：** 已完成

## 项目页的可验证信息

- 目标：通过共享操作 Harness，把云端 GPT-5.4 语义工具调用轨迹蒸馏到本地小模型。
- 演示：MuJoCo 操作、Franka 真机桌面整理与扰动恢复；视频为作者展示，不等同第三方复现。
- 报告：15 项模拟任务成功率 87.1%；9 项真实任务 90.0%；单 RTX 5090 的模型调用延迟均值 0.587s，不能等价于全流程物理操作速度。

## 开放状态

- **已公开：** GitHub 仿真 Harness/任务/轨迹与训练脚本、Hugging Face 学生 checkpoint 和 SFT 轨迹。
- **有条件：** SAM3 感知 checkpoint 等第三方模型需要各自访问权限；真实 FR3 机械臂、相机与平台环境无法由网页自动提供。
- **授权：** 官方仓库根目录未发现顶层 LICENSE 文件，不能仅凭公开 GitHub 即推断全部代码与数据具备某种商业再利用许可；第三方组件另须核验。
- **区分同名：** 本项目 Guava 是机器人操作模型蒸馏，不是 PEAR 论文中用于人体 3DGS 渲染的同名 GUAVA。

## 关联

- [论文精读](../papers/guava_arxiv_2606_18363.md)
- [源码核验](../repos/guava-release.md)
- [知识实体](../../wiki/entities/paper-guava-harness.md)
