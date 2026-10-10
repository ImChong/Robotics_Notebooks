# amazon-far/php_parkour（PHP 官方代码仓库）

- **类型：** repo / humanoid parkour training, motion matching & sim2sim
- **URL：** <https://github.com/amazon-far/php_parkour>
- **许可证：** Apache-2.0
- **配套项目页：** <https://php-parkour.github.io/>
- **论文：** [Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching（arXiv:2602.15827）](https://arxiv.org/abs/2602.15827)
- **核验日期：** 2026-10-10

## 仓库范围

官方代码仓现公开了 PHP 的主要研究与复现流程，不只是项目页或浏览器 demo：

| 路径 / 入口 | 用途 | 运行环境 / 说明 |
|---|---|---|
| motion_matching/ | 生成、播放和检查 G1 跑酷 motion/terrain 数据集 | Conda；指南注明不需要 IsaacSim |
| wbt_training/ | 训练 terrain teacher、DAgger + RL 蒸馏学生、评估与导出 ONNX | Linux/NVIDIA + IsaacSim |
| run_php_sim.sh | 启动带 D435i 深度相机的 MuJoCo 仿真与深度共享内存 | 通过仓库固定的 Holosoma 子模块运行 |
| run_php_inference.sh | 加载 depth backbone + student ONNX，运行策略推理 | 与 MuJoCo 仿真配套 |
| scripts/download_assets.py | 校验并下载发布的示例数据或学生模型 | 检查发布归档与文件校验信息 |

## 数据与模型开放边界

- **训练数据：** 发布 training-motion-examples-v1，包含 locomotion、不同速度 step 与 climb-76 的 motion/terrain 示例包，可跳过 motion matching 直接开始 teacher training。
- **学生模型：** student-assets-v1 发布清理过的 depth_backbone.onnx 和 student.onnx，可按官方 sim2sim 指南在 MuJoCo 中运行。
- **没有发布的部分：** README 明确说明上述模型发布包不包含原始 .pt teacher 或 student checkpoints；示例数据也不是预训练 teacher 权重。
- **复现成本：** 准备数据与 motion matching 相对独立；训练/评估依赖 IsaacSim 和 NVIDIA Linux 环境。仓库的 inference launcher 是 MuJoCo sim2sim 路径，不应把它误读成任意真机部署包。
- **网站 demo：** <https://php-parkour.github.io/demo.html> 是浏览器内 MuJoCo 展示入口，不替代本地训练环境和源码运行链。

## 关键入口

- [README：总览、下载学生模型、快速运行入口](https://github.com/amazon-far/php_parkour#readme)
- [Motion matching 指南](https://github.com/amazon-far/php_parkour/blob/main/motion_matching/README.md)
- [训练与评估指南](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/README.md)
- [MuJoCo sim2sim 指南](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md)
- [学生 ONNX release](https://github.com/amazon-far/php_parkour/releases/tag/student-assets-v1)
- [示例训练数据 release](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1)
- [Apache-2.0 License](https://github.com/amazon-far/php_parkour/blob/main/LICENSE)

## 关联资料

- [PHP 论文归档](../papers/php_parkour_arxiv_2602_15827.md)
- [PHP 项目页归档](../sites/php-parkour-github-io.md)
- [PHP wiki 实体](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)
