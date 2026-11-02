# From RT1 to MolMoAct2: A brief history of VLAs

> 来源归档（X Article / 个人博客）

- **标题：** From RT1 to MolMoAct2: A brief history of VLAs
- **类型：** blog（X Article）
- **作者：** Khushi（@starlitmatcha，简介「learning superduperintelligence for robotics」）
- **原始链接：** https://x.com/starlitmatcha/status/2116918927913804223（推文）；文章 https://x.com/i/article/2116902127171932161
- **发表日期：** 2026-11-01
- **入库日期：** 2026-11-02
- **抓取方式：** X 页面为 JS 壳，用第三方接口 `api.fxtwitter.com` 取推文与 Article 结构化内容（139 个文本块）；原始 JSON 落盘于 [`sources/raw/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.json`](../raw/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.json)；文章内的图片与实验图未取回
- **一句话说明：** 个人学习笔记式长文，沿 SayCan → RT-1 → RT-2 → Open X-Embodiment/RT-X → Octo → OpenVLA → π0 → SmolVLA → MolmoAct / MolmoAct 2 梳理 VLA 架构演进，并附一个 RT-1 离散化的小实验。
- **说明：** 文中数字与论断为作者表述；与论文的核对见各模型节点。文中提到的 GPT-4V、Gemini、LLaVA、PaLI-X、PaLM-E、Prismatic、PaliGemma 等只作背景提及，不单独建节点。

## 文章结构与要点

- **为什么需要 VLA**：机器人在条件稍变（如料筐挪两英寸）就失效；VLM 能理解但不能动，经典控制能动但不泛化。VLA 的洞见：给 VLM 增加动作输出，用机器人示范端到端训练。
- **VLA 如何工作**：相机图像经视觉编码器成视觉 embedding，指令分词成文本 embedding，许多模型还输入关节角、夹爪开度等状态；Transformer 联合处理后由动作头预测末端位移、旋转与开合——可以是离散动作 token，也可以是连续值或 diffusion / flow matching，常一次预测多步动作 chunk；控制器执行后再观测、循环。
- **动作表示的三种路线**：离散动作 token（RT-1、RT-2、OpenVLA）→ 扩散出的连续动作 chunk（Octo）→ VLM + flow matching 动作专家（π0、SmolVLA、MolmoAct 2）。
- **开放问题（作者观点）**：(1) 示范只能教起点，机器人也应从自身错误中学习，VLA 与 RL 是互补而非对立；(2) 把 VLA 与世界模型结合，预测动作后果、比较候选动作；作者承认预测不准与额外计算延迟是风险。

## 模型 → 节点（10 个模型，10 个唯一节点）

| # | 文中名称 | 节点 | 状态 |
|---|----------|------|------|
| 1 | SayCan | [paper-saycan](../../wiki/entities/paper-saycan.md) | 复用 |
| 2 | RT-1 | [paper-rt-1](../../wiki/entities/paper-rt-1.md) | 复用 |
| 3 | RT-2 | [paper-rt-2](../../wiki/entities/paper-rt-2.md) | 复用 |
| 4 | Open X-Embodiment / RT-X | [paper-open-x-embodiment](../../wiki/entities/paper-open-x-embodiment.md) | 复用 |
| 5 | Octo | [paper-octo](../../wiki/entities/paper-octo.md) | 复用 |
| 6 | OpenVLA | [paper-openvla](../../wiki/entities/paper-openvla.md) | 复用 |
| 7 | π0 | [paper-pi0](../../wiki/entities/paper-pi0.md) | 复用 |
| 8 | SmolVLA | [paper-smolvla](../../wiki/entities/paper-smolvla.md) | 新建 |
| 9 | MolmoAct | [paper-molmoact](../../wiki/entities/paper-molmoact.md) | 新建 |
| 10 | MolmoAct 2 | [paper-molmoact2](../../wiki/entities/paper-molmoact2.md) | 复用 |

## 离散化实验（作者表述）

作者取 3 条 LIBERO 示范共 843 个动作，把每个动作维度舍入到 16 / 32 / 64 / 128 / 256 个均匀格点，并比较「覆盖全范围」与「裁到 1–99 百分位」两种量化范围：16 格时裁剪略优；256 格时全范围的重构误差约低 5.5 倍；结论是动作分辨率取决于格点数与覆盖范围两者，且单步更准不保证长时程行为同比例更好。（实验图未取回，数字未复现。）

## 文章内的笔误 / 重复（已按原样处理）

- 作者把 Karol Hausman 写作「Carol Hausman」；OpenVLA 的离散化段落在正文中重复出现两次；这些不影响结论，页面不转录笔误。
- 文章把 MolmoAct 2 称为「更快、更实用」，具体指标文中未给。

## 对 wiki 的映射

- 阅读坐标：[VLA 演进简史：从 SayCan / RT-1 到 MolmoAct 2（Khushi 长文阅读坐标）](../../wiki/overview/vla-history-rt1-to-molmoact-technology-map.md)
