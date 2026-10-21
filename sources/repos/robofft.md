# RoboFFT 官方代码仓库

- **仓库：** https://github.com/shshsunny/RoboFFT
- **默认分支：** `release`
- **关联论文：** [arXiv:2609.32236](https://arxiv.org/abs/2609.32236)
- **项目页：** https://student-of-holmes.github.io/RoboFFT/
- **模型 / 数据：** [Hugging Face](https://huggingface.co/shshsunny/RoboFFT)；[ModelScope](https://modelscope.cn/models/shshsunny/RoboFFT/tree/master)
- **核查日期：** 2026-10-16
- **一句话说明：** RoboFFT-F（flow / ReFlow）与 RoboFFT-D（diffusion）的 Robomimic 在线 RL 微调代码、配置和运行入口。

## 开源范围

已发布部分包括 Robomimic 的四个任务（Lift、Can、Square、Transport）、state 与 pixel 输入实验，以及 RoboFFT-F / RoboFFT-D 和部分基线配置。两条实验分支分别位于 `robofft-f/` 与 `robofft-d/`，入口均为 `script/run.py`。

GitHub README 明确说明，论文中 VLA 和真机 RL 实验没有包含在当前代码 release。项目页列有公开 checkpoint / 数据镜像，但 README 同时提示部分资产下载 URL 仍待配置；开始复现前需检查配置中的 `train_dataset_path`、`normalization_path`、`base_policy_path`。

## 运行入口

- flow：进入 `robofft-f/`，安装其独立环境，按 README 运行 `python script/run.py --config-name=robofft_reflow_mlp --config-dir=cfg/robomimic/finetune/<task>`
- diffusion：进入 `robofft-d/`，安装其独立环境，按 README 运行 `python script/run.py --config-name=robofft_diffusion_mlp --config-dir=cfg/robomimic/finetune/<task>`
- 数据目录、检查点下载及像素输入配置以仓库 README 与各分支配置为准。

- **归档：** [项目页核查](../sites/robofft.md) · [论文摘录](../papers/robofft_arxiv_2609_32236.md)
- **wiki：** [RoboFFT 实体页](../../wiki/entities/paper-robofft.md)
