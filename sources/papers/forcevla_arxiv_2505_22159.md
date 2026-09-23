# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation

> 来源归档（ingest）

- **标题：** ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
- **类型：** paper / vla / force-torque / moe / contact-rich-manipulation / pi0
- **会议：** NeurIPS 2025
- **arXiv abs：** <https://arxiv.org/abs/2505.22159>
- **arXiv HTML：** <https://arxiv.org/html/2505.22159>
- **PDF：** <https://arxiv.org/pdf/2505.22159>
- **项目页：** <https://sites.google.com/view/forcevla2025>
- **机构：** Shanghai Jiao Tong University；Shanghai AI Lab；Fudan University；Shanghai Innovation Institute；Noematrix Intelligence 等
- **通讯作者：** Qiaojun Yu（yqjllxs@alumni.sjtu.edu.cn）
- **入库日期：** 2026-09-23
- **一句话说明：** 在 **π₀** 框架上将 **6 轴外载力/力矩** 提升为一等模态，**FVLMoE** 在 VLM 编码后动态路由融合力 token 与视–语嵌入；**ForceVLA-Data**（5 任务 / 244 轨迹 / 140k 步）上平均成功率 **60.5%**，较 π₀-base w/ force **+23.2 pt**；插头插入等最高 **80%**。

## 开源核查（2026-09-23）

| 项 | 状态 |
|----|------|
| 项目页 | <https://sites.google.com/view/forcevla2025> — 方法图、五任务视频、泛化/消融；**未列** GitHub / HF 下载按钮 |
| 论文承诺 | Abstract / §4.3：**「Code and data will be released at website」**；数据集与采集脚本将公开 |
| 第三方镜像 | 检索到非官方 fork [`tshiamor/ForceVLA`](https://github.com/tshiamor/ForceVLA) 与 HF [`qiaojunyu/ForceVLA-real-data`](https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data) — **非项目页官方入口** |
| 结论 | **待发布**（以论文与官方项目页为准；截至核查日无官方代码链） |

## 摘要级要点

- **问题：** VLA 偏语义/空间规划，接触丰富阶段力需求随相位变化；纯视觉在遮挡/不确定性下 brittle。
- **观测：** $O_t=\{V_t^b, V_t^h, s_t, f_t\}$；$f_t\in\mathbb{R}^6$ 外载 wrench（世界系）；语言 $L$ → 动作块 $A_t$（flow matching）。
- **FVLMoE：** 力经线性投影为 token，与 SigLIP/PaliGemma VLM 输出拼接；**sparse MoE（E=4, top-k=1）** 自适应路由；融合特征注入 flow action head。
- **设计选择：** **力必须在 VLM 之后** 注入；early fusion MoE 可致 **0%**；late concat **60%**；FVLMoE **80%**（plug 消融）。
- **ForceVLA-Data：** Bottle Pumping、Plug/USB Insertion、Whiteboard Wiping、Cucumber Peeling；244 轨迹、140k 同步步；480×640 图像。
- **五任务平均：** ForceVLA **60.5%** vs π₀-base w/ F **37.3%**；擦板分两阶段指标；黄瓜削皮 **14.12 cm/刀**、**7 刀**完成。
- **泛化：** 几何/高度/遮挡/不稳定插座；遮挡设置 **90%**。

## 核心论文摘录（MVP）

### 1) FVLMoE：力作为 VLM 后一等模态

- **链接：** <https://arxiv.org/html/2505.22159#S4>
- **摘录要点：** $E_{in}=[E_{VL}; E_F]$ → encoder → MoE → 残差 → 投影；$G_{\text{FVLMoE}}$ 与 proprio/noisy action suffix **相加** 调制 flow denoising。
- **对 wiki 的映射：**
  - [FWBC-VLA](../../wiki/entities/paper-fwbc-vla.md) — 力觉 VLA 对照（腕部 F/T vs 机身补偿）
  - [FM-VLA](../../wiki/entities/paper-fm-vla.md) — 「力=瞬时条件」族谱

### 2) ForceVLA-Data 与五任务 benchmark

- **链接：** <https://arxiv.org/html/2505.22159#S4.SS3>；项目页 Experimental Setups
- **摘录要点：** 遥操采集 pipeline；五类接触动态差异任务；同步 vision + proprio + F/T。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 3) MoE 路由与相位 specialization

- **链接：** 项目页 Visualization and Case Studies / Figure 6
- **摘录要点：** 不同任务/完成度下 expert load 随时间变化；insert plug / peel 相位 specialist；Expert 0 跨任务通用。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md) — MoE 多模态融合实例

## 对 wiki 的映射（汇总）

- 交叉实体：[FWBC-VLA](../../wiki/entities/paper-fwbc-vla.md)、[FM-VLA](../../wiki/entities/paper-fm-vla.md)、[DeCAL](../../wiki/entities/paper-decal.md)（ForceVLA 类低维力通道对照）
- 概念：[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[VLA](../../wiki/methods/vla.md)
- **注：** 尚无独立 `wiki/entities/paper-forcevla*.md`；后续 ingest 可升格

## 当前提炼状态

- [x] FVLMoE 架构、ForceVLA-Data、五任务/泛化/消融、待发布开源状态已摘录
- [ ] 官方代码发布后需更新开源核查表

## BibTeX

```bibtex
@article{yu2025forcevla,
  title={ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation},
  author={Yu, Jiawen and Liu, Hairuo and Yu, Qiaojun and Ren, Jieji and Hao, Ce and Ding, Haitong and Huang, Guangyu and Huang, Guofan and Song, Yan and Cai, Panpan and Zhang, Wenqiang and Lu, Cewu},
  journal={arXiv preprint arXiv:2505.22159},
  year={2025},
  url={https://arxiv.org/abs/2505.22159},
}
```
