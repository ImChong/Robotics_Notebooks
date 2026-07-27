# Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers（WorldWeaver / W²，arXiv:2607.21594）

> 来源归档（ingest）

- **标题：** Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers
- **短名：** WorldWeaver（\(\mathbf{W}^{\mathbf{2}}\)）
- **类型：** paper / streaming video diffusion / multi-agent world model / world state registers
- **arXiv：** <https://arxiv.org/abs/2607.21594>（HTML：<https://arxiv.org/html/2607.21594v1>）
- **项目页（canonical Pages）：** <https://vail-ucla.github.io/worldweaver/>
- **项目页（机构站镜像）：** <https://vail.cs.ucla.edu/worldweaver/>
- **代码仓：** <https://github.com/VAIL-UCLA/WorldWeaver>（README：**Code and checkpoints are coming soon**）
- **作者：** Sicheng Mo*、Yuheng Li*、Ziyang Leng、Krishna Kumar Singh、Bolei Zhou（* Equal contribution）
- **机构：** 加州大学洛杉矶分校（UCLA）、奥多比研究院（Adobe Research）
- **入库日期：** 2026-07-27
- **一句话说明：** 在流式多智能体自回归视频扩散中引入 **World State Registers（WSR）**：可跨智能体持久、按 chunk 动态更新的寄存器 token；MoT 分路建模状态与帧；用 agent status / BEV / scene text 监督接地；Minecraft 双智能体实验提升逻辑一致性与 world score。

## 开源状态（项目页 + 仓库核查，2026-07-27）

- **宣称将开源 / 占位仓：** [VAIL-UCLA/WorldWeaver](https://github.com/VAIL-UCLA/WorldWeaver) 公开，但 README 明确 **「Code and checkpoints are coming soon」**；项目页有论文与方法说明，**未提供可运行训练/推理入口或权重下载**。截至入库日记为 **宣称将开源**，勿写成已可复现。

## 摘要级要点

- **动机：** 仅靠观测历史 KV 难维持多视角共享世界；镜头外演化需显式状态。
- **WSR：** \(\mathbf{r}_i=G_\theta(\mathbf{r}_{i-1},\mathbf{x}_{i-W+1:i},a_i)\)，再条件 \(p(\mathbf{x}_{i+1}\mid \ldots,\mathbf{r}_i)\)。
- **监督：** agent 位姿速度、BEV（DINOv2 cosine）、scene text CE；训练头推理丢弃。
- **训练：** Stage1 双向多玩家 teacher → Stage2 因果 + register → Stage3 Self-Forcing。
- **结果：** Baseline world score **81.0** → Registers only **93.8** → +All **105.1**（项目页表）。

## 核心论文摘录（MVP）

### 1) World state registers

- **链接：** §3.3；Fig. 2
- **摘录要点：** 寄存器跨智能体持久且逐步更新；交错序列 \([x_0,r_0,x_1,r_1,\ldots]\)；局部窗 \(W\)。
- **对 wiki 的映射：**
  - [WorldWeaver](../../wiki/entities/paper-worldweaver.md)
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) — **持续状态** 输出族。

### 2) MoT + 三阶段课程

- **链接：** §3.1–3.4
- **摘录要点：** 状态与帧分权重、联合注意力；Self-Forcing 暴露寄存器漂移。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 3) 开源边界

- **链接：** GitHub README「Code Release」
- **摘录要点：** coming soon；联系 smo3@cs.ucla.edu。
- **对 wiki 的映射：**
  - [`sources/repos/worldweaver.md`](../repos/worldweaver.md)

## BibTeX

```bibtex
@article{mo2026worldweaver,
  title   = {Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers},
  author  = {Mo, Sicheng and Li, Yuheng and Leng, Ziyang and Singh, Krishna Kumar and Zhou, Bolei},
  journal = {arXiv preprint arXiv:2607.21594},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-worldweaver.md`](../../wiki/entities/paper-worldweaver.md)
- 项目页：[`sources/sites/worldweaver-vail-ucla.md`](../sites/worldweaver-vail-ucla.md)
- 代码占位：[`sources/repos/worldweaver.md`](../repos/worldweaver.md)
- 策展语境：[`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
