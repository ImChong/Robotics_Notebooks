# Möbius: A Unified Motion-Language Model Bridging Understanding and Generation

- **论文：** https://arxiv.org/abs/2611.05279v1
- **版本：** 2026-11-03，v1
- **作者：** Jiazhi Shu 等
- **机构：** 香港科技大学、腾讯游戏 Motus、MIT
- **项目：** https://jiazhishu.github.io/mobius/
- **官方仓库：** https://github.com/JiazhiShu/mobius-code
- **核对范围：** arXiv 摘要及元数据、官方仓库 README；论文全文方法、实验和附录尚未通读。
- **通读：** 未完成

## 已核实摘录

1. 统一处理 motion captioning、text-to-motion、temporal completion、instruction-guided editing。
2. 文本、干净动作、带噪动作参与共享注意力，理解与生成流采用角色专属参数。
3. 直接处理连续帧级动作，无需离散 tokenization 或预训练 motion VAE。
4. 摘要报告 HumanML3D 上统一动作模型中的领先表现，并提及 1.7B 参数与 1700 小时动作数据；具体表格尚待全文核查。

## 对 wiki 的映射

- [Möbius 项目实体](../../wiki/entities/paper-mobius-motion-language.md)

## 待完成

- 通读全文与附录，核验架构、训练目标、实验对比和局限。
- 完成基于论文图表的技术细节补充，执行仓库 CI 与健康度检查。
