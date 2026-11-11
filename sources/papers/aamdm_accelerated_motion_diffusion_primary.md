# AAMDM: Accelerated Auto-regressive Motion Diffusion Model

- **类型：** 作者一手论文
- **作者：** Tianyu Li, Calvin Qiao, Guanqiao Ren, KangKang Yin, Sehoon Ha
- **会议：** CVPR 2024
- **机构：** 佐治亚理工学院（Georgia Tech）；不列颠哥伦比亚大学（UBC）；北京航空航天大学（Beihang University）；西蒙菲莎大学（Simon Fraser University）
- **论文：** https://arxiv.org/abs/2401.06146
- **原始版本：** arXiv:2401.06146（以所列论文链接版本为准）
- **项目页：** https://easypapersniper.github.io/projects/AAMDM/AAMDM.html
- **代码：** 作者主页未列可核实官方代码
- **核对范围：** 核对论文官方页面/HTML 可见的方法与结果、作者项目主页；未核实到可运行实现；全文细节、所有附录未逐段完成交叉核对。
- **通读：** 未完成

## 方法与应用

首先学习人体动作的潜空间表示；DD-GAN 使用少步逆扩散快速产生候选，ADM 用额外扩散步骤修正细节。论文描述配置为生成三步、精修两步；不是与 A-MDM 相同的模型。

## 实验与结论

CVPR 2024 论文在 LaFAN1、交互控制及人造多模态数据上做定量与消融；需区分在线虚拟角色动作与硬件控制延迟。

## 局限 / 对 wiki 的映射

尚无已确认官方源码入口，不能绘制虚构调用时序；论文实验不支持推断 G1 真机运行。

- [AAMDM：快速生成与扩散精修双模块](../../wiki/entities/paper-aamdm-accelerated-motion-diffusion.md)
