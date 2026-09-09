# OpenWAM（系统化世界–动作预训练）

> 来源归档（ingest）

- **标题：** OpenWAM: An Open, Modular Exploration Towards Systematic World–Action Model Pretraining
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.07398>
- **项目页：** <https://openwam-official.github.io/>
- **机构：** 新加坡国立大学（NUS）；清华大学（Tsinghua）；北京大学（PKU）；香港大学（HKU）；浙江大学（ZJU）；香港中文大学（CUHK）；上海交通大学（SJTU）
- **作者：** Yuran Wang、Siqiao Huang、Mingleyang Li、Chenhao Zhang、Jiaqi Liang、Weiyang Jin、Yue Chen、Xuemin Chi、Donghao Zhou、Qize Yu、Yu-Kai Wang、Yuhan Rui、Shenzhe Yao、Zhen Yuan、Zhenhao Shen、Kefei Zhu、Zijie Zhu、Ning Gao、Xiaowei Chi、Guanqi He、Shanghang Zhang、Hao Dong、Lin Shao、Hang Zhao
- **代码：** <https://github.com/OpenWAM-Official/OpenWAM>
- **权重 / 数据：** <https://huggingface.co/OpenWAM>
- **入库日期：** 2026-09-09
- **一句话说明：** 把 WAM 设计空间拆成可组合模块，经六项对照实验提炼三条预训练原则，并发布 OpenWAM-α 全栈（Infra / Study / 预训练模型 + 8 仿真 + 3 真机平台）。

## 核心摘录（MVP）

### 1) 问题：单体 WAM 掩盖设计选择

- **摘录要点：** 现有 WAM 把生成骨干、视觉表征、架构、信息流、推理与训练数据紧耦合，难以判断哪些设计真正重要。OpenWAM 把 world–action 预训练变成 **可控实验程序**：Infra 统一训练 / 推理 / 部署 / 评测；Study 按序回答「继承什么」「世界与动作如何协同」「协同如何跨域缩放」。
- **对 wiki 的映射：**
  - [OpenWAM](../../wiki/entities/paper-openwam.md) — 总览与三层栈。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint WAM 工程化对照坐标。

### 2) OpenWAM-Study 三条原则（Q1–Q6 累积）

- **摘录要点：**
  - **继承（Q1–Q3）：** 足够强的生成骨干 + 紧凑信息丰富的潜空间最能迁移上游世界知识；表征编码器经维度压缩也可与重建式编码器竞争。
  - **协同（Q4–Q6）：** 需要 **专用动作容量**、**显式 world→action 信息流** 与 **同步联合去噪**；默认 **DualSystem joint self-attention + mutual mask + synchronized denoising**。
  - **缩放（数据配方）：** 具身预训练主要扩 **OOD 泛化**；机器人轨迹保动作接地，egocentric 视频扩迁移；**一阶段 ego+robot 共训** 整合二者；预训练规模上 **mutual visibility** 一致更优。
- **对 wiki 的映射：**
  - [OpenWAM](../../wiki/entities/paper-openwam.md) — Study 读法。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 视频骨干与联合去噪。

### 3) OpenWAM-α 架构与数据

- **摘录要点：** **Wan2.2-TI2V-5B** 视频 DiT（width 3072，3D RoPE）+ 专用 **ActionDiT**（width 1024，1D RoPE）；30 层 **joint self-attention** 桥接；**mutual mask**（干净首帧行不看噪声未来与动作）；冻结 **Wan2.2-VAE** 与 **umT5** 语言。预训练 **518.5M 帧（≈6,369 h）**，**70% 机器人 / 30% egocentric 人视频**，**80-D 统一动作空间**。
- **对 wiki 的映射：**
  - [OpenWAM](../../wiki/entities/paper-openwam.md) — 架构表与流程图。
  - [VLA](../../wiki/methods/vla.md) — 与 π₀ / π₀.₅ / Fast-WAM 等基线对照。

### 4) 评测数字（项目页 / 论文摘要）

- **摘录要点：**
  - **LIBERO** 平均成功率 **99.3%**（相对最佳 WAM −0.1、最佳 VLA +0.1）。
  - **RoboTwin2.0-Full** **89.0%**；**LIBERO-plus** **77.1%**；**EBench** **60.5%**；**RoboCasa-GR1** **38.2%**。
  - **RoboDojo** 真机双臂 **37.6 分 / 24.4% SR**（次佳 π₀.₅ 为 22.9 / 12.8%）。
  - **灵巧手 OOD**：预训练混合未出现该本体，仍报告 in-distribution 与背景 / 布局 / 光照 / 物体 OOD 变体。
  - **8 个仿真基准 + 3 个真机平台**；HF 发布 **46** 个检查点。
- **对 wiki 的映射：**
  - [OpenWAM](../../wiki/entities/paper-openwam.md) — 结论与局限。

### 5) 开源状态（截至 2026-09-09，项目页核查）

- **摘录要点：** **已开源**。GitHub `OpenWAM-Official/OpenWAM` 含 Hydra 配置、`openwam/train`、`openwam/deploy` WebSocket 策略服务、**8+ 基准评测客户端**（LIBERO / LIBERO-plus / RoboTwin / RoboCasa365 / RoboCasa GR1 / VLABench / EBench / RoboDojo 等）；HF `OpenWAM` 托管预训练与微调权重；已接入 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)。
- **对 wiki 的映射：**
  - [openwam 仓库](../repos/openwam.md)
  - [OpenWAM 项目页](../sites/openwam-official.md)

## 当前提炼状态

- [x] arXiv 摘要、项目页与 README 已对齐摘录
- [x] 仓库 / HF / 项目页已交叉核查（**已开源**）
- [x] wiki 映射：`wiki/entities/paper-openwam.md` 新建
