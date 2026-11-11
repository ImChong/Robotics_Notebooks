# AAMDM: Accelerated Auto-regressive Motion Diffusion Model

- **类型：** 作者一手论文
- **作者：** Tianyu Li, Calvin Qiao, Guanqiao Ren, KangKang Yin, Sehoon Ha
- **会议：** CVPR 2024
- **机构：** 佐治亚理工学院（Georgia Tech）；不列颠哥伦比亚大学（UBC）；北京航空航天大学（Beihang University）；西蒙菲莎大学（Simon Fraser University）
- **论文：** https://arxiv.org/abs/2401.06146
- **原始版本：** arXiv:2401.06146（以所列论文链接版本为准）
- **项目页：** https://easypapersniper.github.io/projects/AAMDM/AAMDM.html
- **代码：** 作者主页未列可核实官方代码
- **核对范围：** arXiv:2401.06146v1 全文：§1–2 背景与相关研究、§3 编码潜空间/ADM 精修/DD-GAN 三步生成/梯度约束、§4 LaFAN1 实验及表1–2 消融、§5 局限、§6 Supplementary Material 训练算法伪代码和全部超参数；参考文献浏览。另核对 CVPR 2024 作者官方出版页。
- **通读：** 已完成

## 方法与应用

首先学习人体动作的潜空间表示；DD-GAN 使用少步逆扩散快速产生候选，ADM 用额外扩散步骤修正细节。论文描述配置为生成三步、精修两步；不是与 A-MDM 相同的模型。

## 实验与结论

CVPR 2024 论文在 LaFAN1、交互控制及人造多模态数据上做定量与消融；需区分在线虚拟角色动作与硬件控制延迟。

## 局限 / 对 wiki 的映射

尚无已确认官方源码入口，不能绘制虚构调用时序；论文实验不支持推断 G1 真机运行。

- [AAMDM：快速生成与扩散精修双模块](../../wiki/entities/paper-aamdm-accelerated-motion-diffusion.md)

## 原文复核摘录（正文证据）
- **生成序列：** Autoencoder 把 338 维 pose 与未来路线特征映射到 64 维 embedding（52 learned latent + 12 规划特征）；DD-GAN 生成 3 步 + ADM 精修 2 步，共 5 步，输出下一帧后迭代。
- **实验：** LaFAN1 25 段、约 100k 帧转换、26.67 分钟；表1 AAMDM 173 FPS、AMDM200 4.72 FPS，方向控制误差 TE-UC=0.034；表2 无 polishing 出现长序列失稳，去掉 embedding FID 显著上升。
- **附录：** §6 的 Algorithm 1 明确先训练/滚动精修 ADM，再通过 DD-GAN G/D 优化快速生成分支；训练窗口 10，batch 64，ADM LR 3e-4，G/D LR 1e-4。
- **参考：** §3.1–§3.6、§4.1–§4.3、§5、§6；[论文原文](https://arxiv.org/html/2401.06146)。
