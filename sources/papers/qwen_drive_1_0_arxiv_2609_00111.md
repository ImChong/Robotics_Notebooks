# Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving

> 来源归档（paper）

- **标题：** Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving
- **类型：** paper / autonomous-driving / vision-language / 3d-perception / motion-planning
- **arXiv：** [2609.00111](https://arxiv.org/abs/2609.00111)（v1，2026-08-31）
- **作者：** Xin Zhou、Zongchuang Zhao、Zhibo Yang、Mingsheng Li、Humen Zhong、Shuai Bai、Du Chu、Ruizhe Chen、Zhaohai Li、Jun Tang、Qiuyue Wang、Mingkun Yang、Jiazhao Zhang、Dayiheng Liu、Dingkang Liang、Xiang Bai
- **团队 / 机构：** Qwen Team；华中科技大学（HUST）
- **官方项目文章：** <https://qwen.ai/blog?id=qwen-drive-1.0>
- **代码：** <https://github.com/QwenLM/Qwen-Drive-1.0>
- **权重：** <https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B>
- **入库日期：** 2026-10-14
- **一句话说明：** 在 Qwen3.5-4B 上保留预训练 VLM 架构，外接 BEV 感知头与 flow-matching Planning Expert，统一 3D 感知、驾驶问答和未来轨迹规划。

## 摘要级要点

- **共享基座：** 同一表示服务于驾驶 / 通用 VQA、BEV 3D 感知和轨迹规划。
- **显式空间接口：** BEV 感知头联合预测 3D 目标、语义占用和 BEV 地图。
- **生成式规划：** Planning Expert 条件于共享表征，通过 flow matching 生成未来自车轨迹。
- **训练数据：** 分阶段训练结合驾驶监督与通用 VLM 数据，并统一跨数据集标签、VQA 回答和轨迹表示。
- **边界：** 官方后续说明文本推理与生成轨迹的一致性仍待提升。

## 核心论文摘录（MVP）

### 1) 共享 VLM + 两个外接任务模块

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2609.00111)；官方博客「Model Architecture」
- **摘录要点：** 保留预训练 VLM 架构，在共享表征上外接 BEV 感知头和 Planning Expert。BEV 头联合做 3D 检测、语义占用与地图分割；规划专家生成未来自车轨迹。
- **对 wiki 的映射：**
  - [Qwen-Drive-1.0 项目实体](../../wiki/entities/paper-qwen-drive-1-0.md)
  - [端到端自动驾驶方法地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 2) 数据规范化与分阶段训练

- **链接：** 官方博客「Highlights」和「Motion Planning」
- **摘录要点：** 管线统一跨数据集感知标签、整理驾驶 VQA 格式与事实一致性、规范轨迹 waypoint；官方报告规划训练包含 2.83M 公共样本，并按 10 Hz 预测 5 秒轨迹。
- **对 wiki 的映射：**
  - [Qwen-Drive-1.0 项目实体](../../wiki/entities/paper-qwen-drive-1-0.md)
  - [VLA 方法页](../../wiki/methods/vla.md)
  - [端到端自动驾驶方法地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)

### 3) 多协议评测与可复现边界

- **链接：** 官方博客「Performance」；[官方代码 README](https://github.com/QwenLM/Qwen-Drive-1.0)
- **摘录要点：** NAVSIM v1.1 navtest 的 PDMS 为 SFT 88.2 / RL 90.7，best-of-6 为 89.3 / 91.4；Waymo E2E RFS 为 7.78 / 7.91；PhysicalAI 3 秒 minADE 为 SFT 0.34 m、RL 0.38 m。官方仓库有推理 / 评测脚本与权重入口，但未发现训练脚本目录。
- **对 wiki 的映射：**
  - [Qwen-Drive-1.0 项目实体](../../wiki/entities/paper-qwen-drive-1-0.md)
  - [端到端自动驾驶方法地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)
  - [S²-VLA 驾驶规划对照](../../wiki/entities/paper-s-squared-vla.md)

## 对 wiki 的映射

- 主项目实体：wiki/entities/paper-qwen-drive-1-0.md
- 配套项目文章：sources/sites/qwen-drive-1-0.md
- 配套代码仓库：sources/repos/qwenlm_qwen_drive_1_0.md
- 互链：[VLA](../../wiki/methods/vla.md)、[S²-VLA](../../wiki/entities/paper-s-squared-vla.md)、[BIRDriver](../../wiki/entities/paper-birdriver.md)、[E2E 自动驾驶地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)

## 参考来源

- arXiv 摘要与版本信息：<https://arxiv.org/abs/2609.00111>
- Qwen 官方文章：<https://qwen.ai/blog?id=qwen-drive-1.0>
- 官方 README 与推理代码：<https://github.com/QwenLM/Qwen-Drive-1.0>
- ModelScope 权重入口：<https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B>
