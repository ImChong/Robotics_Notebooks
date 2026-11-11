# A-MDM: Interactive Character Control with Auto-Regressive Motion Diffusion Models

- **类型：** 作者一手论文
- **作者：** Yi Shi, Jingbo Wang, Xuekun Jiang, Bingkun Lin, Bo Dai, Xue Bin Peng
- **会议：** ACM TOG / SIGGRAPH 2024
- **机构：** 论文作者来自多个机构；本次以原始作者表为准，不推断单一归属
- **论文：** https://arxiv.org/abs/2306.00416v4
- **原始版本：** arXiv:2306.00416（以所列论文链接版本为准）
- **项目页：** https://yi-shi94.github.io/amdm_page/
- **代码：** https://github.com/Yi-Shi94/AMDM
- **核对范围：** arXiv:2306.00416v4 全文：§1 背景、§2 相关研究、§3 自回归 DDPM 与 scheduled sampling/10层 MLP、§4 随机/目标采样/inpainting、§5 PPO 逐去噪残差控制、§6 到达/摇杆/路径任务、§7 LaFAN1/100STYLE/AMASS 实验与速度/质量表、§8 局限；参考文献表浏览。该 HTML 版无独立附录。
- **通读：** 已完成

## 方法与应用

A-MDM 根据历史姿态逐步采样下一帧，用轻量 MLP 条件扩散而非一次生成全时长块；交互目标由任务采样、运动补全以及可训练高层控制接口影响生成。

## 实验与结论

原文比较长时连续生成、交互可控性和动作多样性；官方代码提供 LAFAN1 / 100STYLE 预训练权重和交互运行方式。

## 局限 / 对 wiki 的映射

实时角色动作合成不直接等于物理 G1 控制；不应与 CVPR 2024 的 AAMDM 混为同一论文。

- [A-MDM：自回归扩散角色交互控制](../../wiki/entities/paper-a-mdm-interactive-character-control.md)

## 原文复核摘录（正文证据）
- **关键机制：** 每次条件为上一帧，逆扩散生成下一帧，scheduled sampling 用自身生成状态代替 GT 抑制自回归漂移；10 层、每层 1024 单元 MLP 与扩散步数在算力和质量间折中。
- **控制：** 四套机制是随机采样、基于任务得分的候选采样、扩散过程 feature-mask inpainting、PPO 高层策略输出各扩散步残差；路径任务通过未来 waypoints 训练高层。
- **速度/质量：** 表3 的 100STYLE 40 步单帧约 21ms（RTX 4090），论文明确指出 space-time 扩散 FID 更优，A-MDM 换取低时延；§8 承认不自然约束时脚滑和抖动。
- **参考：** §3–§8；[论文原文](https://arxiv.org/html/2306.00416)。
