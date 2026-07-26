# 𝒩₀-TWAM：Scaling Tactile-Native World Action Model for Contact-Rich Manipulation

> 来源归档（ingest）

- **标题：** N0-TWAM: Scaling Tactile-Native World Action Model for Contact-Rich Manipulation
- **类型：** paper / technical-report / world-action-model / tactile / contact-rich / flow-matching
- **项目页：** <https://research.neoteai.com/n0-twam/>
- **PDF：** <https://research.neoteai.com/assets/n0-twam-report.pdf>
- **代码 / Checkpoints：** <https://github.com/neoteai/N0-TWAM>（截至入库日占位）
- **机构：** NeoteAI × Fudan TEAI
- **日期：** 2026-07-25
- **入库日期：** 2026-07-26
- **一句话说明：** 触觉原生世界–动作模型：非对称 MoT 先联合生成未来视频与未来触觉，再从该多模态未来去噪动作；预测通路（VAE latent）+ 观测通路（NeoForce 力场）双角色触觉。

## 摘要级要点

- **问题：** VLA 不显式建模下一时刻；现有 WAM 多只预测视频；二者都未把接触当作策略必须预见的未来一部分。
- **架构：** 视频专家 5B（宽 3072）+ 动作/触觉专家（宽 1024）；共享全宽 self-attention；约 **7.16B** 可训参数（约三全宽专家一半）；推理缓存预测视频/触觉 K/V，动作步只跑瘦动作专家。
- **双通路触觉：** 预测通路：触觉视频同冻结视频 VAE + latent flow-matching，残差相对初始触觉帧；观测通路：InTac S1→三轴力图→NeoForce token，零初始化 cross-attn 注入动作头前。预训练 **只开预测通路**，后训练再开观测条件。
- **数据：** NeoData 六本体 450 任务 ≈ **7.5M** clips；20 维双手末端动作；128×H800、30k steps；后训练语言/触觉各 10% dropout。
- **主结果：** UniVTAC **84.5%** · NeoSim **49.4%** · 真机八任务 **46.3%**（π₀.₅ 30.0% / LingBot-VA 21.9% / FastWAM 14.4%）；泛化均分 **51.7%**；消融去预测触觉 56.4%、去观测触觉 50.0%（相对满配 67.0% 任务均）。
- **开源状态：** **部分 / 待发布** — roadmap **By July 31, 2026**；仓内暂无可运行实现。

## 核心论文摘录（MVP）

### 1) Predict-then-act 因果级联

- **摘录要点：** frame-id 因果：视频与触觉专家共生成 → 动作专家条件于刚预测的多模态未来；共享 attention mask 内实现。
- **对 wiki 的映射：** [world-action-models.md](../../wiki/concepts/world-action-models.md)、[paper-n0-twam.md](../../wiki/entities/paper-n0-twam.md)

### 2) 预测 vs 观测触觉角色分离

- **摘录要点：** 大规模预训练禁用观测触觉捷径，迫使学接触动力学；仿真用同一条件接口接模拟触觉。
- **对 wiki 的映射：** [visuo-tactile-fusion.md](../../wiki/concepts/visuo-tactile-fusion.md)、[paper-vt-wam-visuotactile-contact-rich.md](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)

### 3) Tactile punctuation（长程分段）

- **摘录要点：** 接触事件切分长演示；部署时预测触觉预见转换、观测触觉确认完成再推进调度。
- **对 wiki 的映射：** [contact-rich-manipulation.md](../../wiki/concepts/contact-rich-manipulation.md)

### 4) NeoSim / UniVTAC / 真机与消融

- **摘录要点：** NeoSim 双手子集大幅领先；数据缩至 20% UniVTAC 降至 65.4%（满配 84.5%）。
- **对 wiki 的映射：** [generative-world-models.md](../../wiki/methods/generative-world-models.md)、[paper-n0-foundation.md](../../wiki/entities/paper-n0-foundation.md)

## 对 wiki 的映射（汇总）

- 实体：[paper-n0-twam.md](../../wiki/entities/paper-n0-twam.md)
- 交叉：[VT-WAM](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md) · [paper-n0-vtla.md](../../wiki/entities/paper-n0-vtla.md) · [neoteai.md](../../wiki/entities/neoteai.md)
- 站点 / 仓：[research-neoteai-com.md](../sites/research-neoteai-com.md) · [n0-twam.md](../repos/n0-twam.md)

## BibTeX

```bibtex
@article{n0twam2026,
  title   = {N0-TWAM: Scaling Tactile-Native World Action Model for Contact-Rich Manipulation},
  author  = {NeoteAI Team and Fudan TEAI Team},
  journal = {Technical Report},
  year    = {2026}
}
```
