# DiT-Block Policy 项目页（dit-policy.github.io）

- **URL：** <https://dit-policy.github.io/>
- **论文：** [robotic_dit_ingredients_arxiv_2410_10088.md](../papers/robotic_dit_ingredients_arxiv_2410_10088.md)
- **代码：** <https://github.com/SudeepDasari/dit-policy> — [`sources/repos/sudeepdasari_dit_policy.md`](../repos/sudeepdasari_dit_policy.md)
- **数据：** <https://huggingface.co/datasets/oier-mees/BiPlay>
- **入库日期：** 2026-09-28
- **最近复核：** 2026-11-19
- **核对范围：** 项目页完整静态文本（`dit_policy_site.txt`）与论文/仓库对照

## 一句话说明

*The Ingredients for Robotic Diffusion Transformers* 官方项目页：展示 DiT-Block Policy 架构细节、ALOHA/Franka 真机长时域任务视频及 BiPlay（7,023 clips，~10h）数据集入口。

## 开源与口径核查（2026-11-19）

- **代码：** 项目页直接链接 **GitHub `SudeepDasari/dit-policy`** → **已开源**。
- **数据：** **BiPlay** 托管于 Hugging Face Datasets（7,023 个未标注双臂 play 片段，约 10 小时）。
- **权重/特征：** README 指向 CMU data4robotics release 与 `download_features.sh`。
- **口径差异提示：** 项目页架构图说明写使用 **ResNet-18** tokenizer，论文正文写 **ResNet-26**，代码配置默认提供 `resnet_gn` / `resnet_gn_nopool`（见 `download_features.sh` 中 `IN_1M_resnet18.pth` 与 `experiments/agent/diffusion.yaml`）。
