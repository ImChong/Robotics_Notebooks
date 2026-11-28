# EponaX：真实道路世界动作模型的训练与部署 / Recipes for Training and Deploying World-Action Models in the Real World

- **类型：** technical report
- **作者：** Mingxiao Li、Zhijian Shu、Wei Yin、Jiawei Xu、Mingkai Jia、Alan Liang、Zhizhou Zhong（共同一作），Shiyao Li、Dongyue Lu、Liang Yue、Zeyuan Xu、Kaicheng Zhang、Xiao-Xiao Long、Lingdong Kong、Qian Zhang。
- **团队：** Horizon Robotics World Model Team 与合作者。
- **原文：** [官方技术报告 PDF](https://garlicba.github.io/EponaX/paper/EponaX_Tech_Report.pdf) · [项目页归档](../sites/eponax-horizon-robotics.md)
- **核对范围：** 官方 PDF 31 页，正文 §1–7、图 1–10、表 1–12、补充图 S1–S5 与引文；对照项目页。报告未标版本号，以 2026-11-28 所取文件为准，SHA-256 为 `4aa68d0afd5c8607ffb888cf425e502d77fd3c82400d23f4143b3fcc7a6c66ed`。未运行模型或实车。
- **通读：** 已完成
- **入库日期：** 2026-11-28

## 可追溯证据

1. **§3、图 2–4：** 六路相机用冻结 DINO-Tok；前视 2048→512，环视合计 10240→640，再加 9 个导航与 2 个边界 token，每帧共 1163 个位置。TrajLoop 从共享骨干各层读取最新帧特征，逐层条件在 5 步流匹配中迭代；车端冻结解码器将 10 个动作潜 token 转为 60 个轨迹点。
2. **§4、表 1–4：** 先做环视压缩器和导航对齐预训练，再用约 3.5 万小时驾驶数据联合训练动作、未来视觉、深度和语义；NAVSIM 做监督微调与 Flow-GRPO。奖励训练每组采样多个候选，不等于评测时多轨迹挑优。
3. **§5、表 5–12：** NAVSIM v1 navtest 91.9→92.5 PDMS，v2 navtest 91.8 EPDMS，v2 navhard 两阶段合分 48.3。实车四条路各三圈，约 240 km，总人工干预 64 次，安全相关子集 12 次；R1 在该子集为 10.3 次/100 km，不能把标题中 3–5 写成每条路线的范围。

- **对 wiki 的映射：** [EponaX](../../wiki/entities/paper-eponax.md)
