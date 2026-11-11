# A-MDM: Interactive Character Control with Auto-Regressive Motion Diffusion Models

- **类型：** 作者一手论文
- **作者：** Yi Shi, Jingbo Wang, Xuekun Jiang, Bingkun Lin, Bo Dai, Xue Bin Peng
- **会议：** ACM TOG / SIGGRAPH 2024
- **机构：** 论文作者来自多个机构；本次以原始作者表为准，不推断单一归属
- **论文：** https://arxiv.org/abs/2306.00416v4
- **原始版本：** arXiv:2306.00416（以所列论文链接版本为准）
- **项目页：** https://yi-shi94.github.io/amdm_page/
- **代码：** https://github.com/Yi-Shi94/AMDM
- **核对范围：** 核对论文官方页面/HTML 可见的方法与结果、作者项目主页；并核对仓库 README 与部分运行入口；全文细节、所有附录未逐段完成交叉核对。
- **通读：** 未完成

## 方法与应用

A-MDM 根据历史姿态逐步采样下一帧，用轻量 MLP 条件扩散而非一次生成全时长块；交互目标由任务采样、运动补全以及可训练高层控制接口影响生成。

## 实验与结论

原文比较长时连续生成、交互可控性和动作多样性；官方代码提供 LAFAN1 / 100STYLE 预训练权重和交互运行方式。

## 局限 / 对 wiki 的映射

实时角色动作合成不直接等于物理 G1 控制；不应与 CVPR 2024 的 AAMDM 混为同一论文。

- [A-MDM：自回归扩散角色交互控制](../../wiki/entities/paper-a-mdm-interactive-character-control.md)
