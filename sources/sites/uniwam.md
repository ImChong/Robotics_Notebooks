# UniWAM 官方项目页与开放资源核查

> 来源归档（site；核查日期：2026-10-13）

- **项目页：** <https://uniwam.github.io/>
- **论文：** <https://arxiv.org/abs/2610.02054> · [HTML v1](https://arxiv.org/html/2610.02054v1)
- **Code：** <https://github.com/UniWAM/UniWAM>（Apache-2.0）
- **ModelScope：** <https://www.modelscope.cn/collections/Kosmos524/UniWAM>
- **Hugging Face：** <https://huggingface.co/collections/chenpyyy/uniwam>
- **开放状态：** 代码已开放，权重通过模型集合提供入口；实际模型文件和许可本次未独立验证。属于部分开放，而非数据、权重、训练和真机部署材料全部齐备。

## 项目页核查结果

项目页本身直接列出 Paper、Code、ModelScope 与 Hugging Face 按钮，并介绍 8B MoT 的 Physical Reasoner、World Generator、Action Predictor 三个专家。训练部分区分多源预训练、未来视觉噪声增强与历史条件 flow matching 后训练；项目页还汇总 LIBERO、LIBERO-Plus、RoboTwin 和双 Piper 真机实验结果。

- 代码按钮指向官方仓库 [`UniWAM/UniWAM`](https://github.com/UniWAM/UniWAM)。用户原始消息中的 `UniWAM/UniWAMModelScope` 地址在本次检查时返回 404，因此知识页统一使用项目页确认的仓库链接。
- 官方 README 给出 ModelScope 集合中的两个 UniWAM 权重名称：`UniWAM-base`（混合机器人、人类和 VQA 数据的预训练 checkpoint）与 `UniWAM-robotwin-clean`（RoboTwin 2.0 clean 子集的后训练 checkpoint）。ModelScope 集合入口可确认，但其当前文件清单、文件版本及授权条款未独立读取。
- README 和指南可核验 RoboTwin 数据转换、训练、策略部署与评估，以及 LIBERO / LIBERO-Plus 训练评估流程；README 明确将 Bridge、DROID、Fractal 与 real-world inference 排除在本次 release 范围之外。
- 机器人及人类来源数据在论文中被用于训练，但并不等同于官方仓库重新分发了这些原始数据。复现需按相应数据集许可另行获取输入数据和基础模型。

## 来源映射

- [UniWAM 论文归档](../papers/uniwam_arxiv_2610_02054.md)
- [UniWAM 官方代码仓库核查](../repos/uniwam.md)
- [UniWAM wiki 实体](../../wiki/entities/paper-uniwam-unified-world-action-model.md)
