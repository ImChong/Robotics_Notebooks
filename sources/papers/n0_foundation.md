# 𝒩₀-Foundation：Towards the Age of Tactile Intelligence

> 来源归档（ingest）

- **标题：** N0-Foundation: Towards the Age of Tactile Intelligence
- **类型：** paper / technical-report / tactile / dataset / foundation / contact-rich
- **项目页：** <https://research.neoteai.com/n0-foundation/>
- **PDF：** <https://research.neoteai.com/assets/n0-foundation-report.pdf>（同目录亦有 `n0-foundation-paper.pdf`）
- **代码：** <https://github.com/neoteai/N0-Foundation>
- **数据集：** <https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData>
- **机构：** 新智具身智能（NeoteAI）；复旦大学可信具身智能研究院（TEAI）
- **日期：** 2026-07-25（项目页）
- **入库日期：** 2026-07-26
- **一句话说明：** 触觉中心具身操作基础栈：耐用视触觉硬件 + 𝒩₀-TacUMI、**NeoData（>30k h）**、**NeoForce 三轴力场表征**、**NeoReal / NeoSim** 评测；开源子集 **OpenNeoData 5k h**。

## 摘要级要点

- **问题：** 大规模机器人学习仍以视觉为主，但插入、抓取、折叠、擦拭等成败由局部力、摩擦、初滑、形变与接触切换决定——相机弱可观。
- **贡献四件套：** (1) 耐用相机式视触觉传感器 + 𝒩₀-TacUMI 同步采集；(2) NeoData 跨本体语料；(3) NeoForce 硬件无关力场表征；(4) NeoReal（真机 10 任务）与 NeoSim（仿真 12 任务）对齐评测。
- **NeoData 规模：** >30,000 h、1.4M episodes、3.3B timesteps、8B RGB / 10B tactile frames；六本体（TacUMI + Franka FR3 / Piper / ARX X5 / UR5e / Flexiv Rizon 4s）；450+ 任务；约 100 名操作员；TacUMI 贡献约 **57%** 轨迹。
- **OpenNeoData：** **5,000 h** 开源子集，六本体、250+ 任务、200+ skills；LeRobot v3.0；HF + ModelScope；门禁 + CC-BY-NC-SA-4.0。
- **NeoForce：** 观测映射为传感面稠密三轴力场（两切向剪切 + 法向压力）；共享 Transformer 融合 RGB–力场 chunk；重建 + 教师引导 latent 预测 + 跨模态对齐。
- **触觉接入对照（固定 π₀.₅）：** 无触觉 26.5%/38.1 → 图像拼接 27.5%/41.4 → action-expert 条件 30.0%/44.3 → **NeoForce 32.5%/47.5**（成功率 / progressive）。
- **开源状态（2026-07-26）：** **部分开源** — OpenNeoData 已放；NeoForce 代码/权重 roadmap **By July 31, 2026**；GitHub 仓目前仅 README/图。

## 核心论文摘录（MVP）

### 1) 触觉基础设施：传感器 + TacUMI

- **链接：** <https://research.neoteai.com/n0-foundation/>
- **摘录要点：** 耐磨玻璃保护层 + 弹性纹理传感层 + 嵌入 RGB；𝒩₀-TacUMI = 双触觉指 + 160° 鱼眼腕部相机 + 低漂 IR 6-DoF 追踪 + 磁吸夹爪开度，约定对齐五台仪器化机器人平台。
- **对 wiki 的映射：**
  - [𝒩₀-Foundation](../../wiki/entities/paper-n0-foundation.md)、[NeoteAI](../../wiki/entities/neoteai.md)

### 2) NeoData 四层任务层级与质检

- **摘录要点：** VLM 提任务模板 → 人核 → 信号锚定边界；L3 完整任务 / L2 子任务 / L1 动作 / L0 原子段；流完整性、运动质量、任务完成、视频有效性、机器人可迁移性检查。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)

### 3) NeoForce：设备无关力场表征

- **摘录要点：** 对抗光学凝胶 / 电容 / 压阻碎片化，统一到三轴力场；力场条件优于原始触觉图像拼接。
- **对 wiki 的映射：**
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 4) NeoReal / NeoSim 与策略对照

- **摘录要点：** 纯视觉最强策略仍留大空间；NeoSim 上 π₀.₅ 45.8%、LingBot-VA 32.1%，双手持续互接触是主瓶颈。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md)、[𝒩₀-VTLA](../../wiki/entities/paper-n0-vtla.md)、[𝒩₀-TWAM](../../wiki/entities/paper-n0-twam.md)

## 对 wiki 的映射（汇总）

- 实体：[paper-n0-foundation.md](../../wiki/entities/paper-n0-foundation.md) · [neoteai.md](../../wiki/entities/neoteai.md)
- 交叉：[visuo-tactile-fusion.md](../../wiki/concepts/visuo-tactile-fusion.md) · [contact-rich-manipulation.md](../../wiki/concepts/contact-rich-manipulation.md) · [topic-tactile.md](../../wiki/overview/topic-tactile.md)
- 站点 / 仓：[research-neoteai-com.md](../sites/research-neoteai-com.md) · [n0-foundation.md](../repos/n0-foundation.md)

## 当前提炼状态

- [x] 四组件、数据规模、OpenNeoData、NeoForce 对照表、开源边界已摘录
- [x] 与 research / company / repos 互证

## BibTeX

```bibtex
@article{n0foundation2026,
  title   = {N0-Foundation: Towards the Age of Tactile Intelligence},
  author  = {NeoteAI Team and Fudan TEAI Team},
  journal = {Technical Report},
  year    = {2026}
}
```
