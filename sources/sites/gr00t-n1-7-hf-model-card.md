# GR00T N1.7 模型卡与 LeRobot 文档（来源归档）

> 来源归档（site）。**GR00T N1.7 没有独立论文/技术报告**；官方模型卡把架构细节指向 GR00T N1 白皮书 arXiv:2503.14734。

- **模型卡：** <https://huggingface.co/nvidia/GR00T-N1.7-3B>（2026-11-05 读取 raw README）
- **代码：** <https://github.com/NVIDIA/Isaac-GR00T>（归档见 [isaac_gr00t.md](../repos/isaac_gr00t.md)）
- **LeRobot 文档：** `docs/source/groot.mdx`、`src/lerobot/policies/groot/README.md`（固定提交 `9e5ee9859480e425a97a6836bb3fee479539beb5`，见 [lerobot 仓库归档](../repos/lerobot.md)）
- **入库日期：** 2026-11-05

## 核心摘录

- 模型卡称：N1.7 是面向通用人形机器人推理与技能的开放基础模型，跨本体，输入语言、图像（及本体状态），用 flow matching action transformer 生成动作 chunk；N1.X 架构细节见 GROOT N1 White Paper（arXiv:2503.14734）。
- 模型卡称 N1.7 家族含 4 个后训练模型：SimplerEnv Bridge、SimplerEnv Fractal、DROID、LIBERO；参数量写 3,000,000,000；许可 NVIDIA Open Model License。
- 模型卡称 VLM 骨干为 Cosmos-Reason2-2B（但卡内「网络架构」段仍写 SigLip2 + T5，属沿用旧描述，与 LeRobot 源码的 Qwen3-VL 骨干不一致；卡内 Cosmos-Reason2-2B 的链接实指向 8B 页）。
- LeRobot 文档称：仅支持 N1.7，N1.5 已移除（旧版用 `lerobot==0.5.1`）；骨干 Cosmos-Reason2/Qwen3-VL；基座 `nvidia/GR00T-N1.7-3B`。
- web 搜索（2026-11-05）未找到 N1.7 专属 arXiv 报告。
- 模型卡「Evaluation」段（只列白皮书定义的基准，无 N1.7 数字）与「System Requirements and Performance」推理延迟表；Isaac-GR00T 主分支 README「What's New in GR00T N1.7 / Key Changes from N1.6」、`examples/SimplerEnv/README.md`（N1.6 对 N1.7 逐任务表）、`examples/LIBERO/README.md`（四套件成功率，括号百分比与次数不符）；另用 AIRoA-0（arXiv:2611.03815 §2.1、表 1）的第三方基线结果：用于实体页「评测结果」与「与其他工作对比」（2026-11-05 补）。

## 对 wiki 的映射

- [paper-gr00t-n1-7](../../wiki/entities/paper-gr00t-n1-7.md)
