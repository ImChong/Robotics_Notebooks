# Wh0 项目页

> 来源归档

- **标题：** Wh0: Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data
- **类型：** site（论文项目页）
- **链接：** <https://chenyt31.github.io/wh0.github.io/>
- **论文：** <https://arxiv.org/abs/2606.22136>
- **代码：** <https://github.com/chenyt31/Wh0>
- **入库日期：** 2026-09-06
- **状态：** Under Review（页眉）
- **一句话说明：** 展示 **WM-H 50k** 合成管线、Co-FT 数据配比、Unitree G1 零样本灵巧操纵 demo 与 **38.9%** 成功率对比表。
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md`](../../wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **代码** | 页内链 **GitHub chenyt31/Wh0** → **已开源** |
| **WM-H 50k 数据** | 页强调 **compute-driven scaling**；未列独立 HF/Zenodo 50k 包（复现靠仓内生成脚本） |
| **视频 demo** | 项目页托管 rollout 与数据集样例 GIF/视频 |

## 页面要点（2026-09-06）

###  headline 数字

| 指标 | 数值 |
|------|------|
| WM-H episodes | **50k** |
| 零样本提升 | **4.7×**（相对 VITRA+teleop 8.3%） |
| 真机任务 | **18** |
| Wh0 成功率 | **38.9%** |
| 生成成本 | **5.44 GPU-h / 1k videos** |

### WM-H 六步管线（交互页）

1. **Instruction** — 双 agent 词表发现 + 平衡采样  
2. **Scene Edit** — 机器人工作区 capture + Qwen-Image-Edit 插物体  
3. **Video Gen** — Wan-I2V · 4-step LightX2V + Qwen3-VL prompt  
4. **Embodiment** — 可选 Qwen-Image-Edit 人手→机器人手  
5. **Motion** — HaWoR → MANO 3D pose  
6. **WM-H** — 带语言 + 3D 手姿的 ego 操纵 episode  

### 消融变体（页内 dataset 卡片）

| 变体 | 差异 |
|------|------|
| **WM-H** | 完整场景 + 人手视频；Co-FT **68%** batch |
| **WM-H w/o Scene Alignment** | 初始帧来自 Ego4D 而非机器人工作区 |
| **WM-H w/ Embodiment Alignment** | 稀疏帧机器人手外观；Co-FT **4%** batch |

### 策略与数据配比

- VITRA-style policy：PaliGemma + MANO 空间扩散动作  
- Co-FT：**28% teleop · 68% WM-H · 4% WM-H EA**  
- 真机：**Unitree G1 + Inspire**；Vision Pro 遥操作；头戴 ego 相机  

## 对 wiki 的映射

- 实体页：[paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md](../../wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md)
- 论文摘录：[wh0_arxiv_2606_22136.md](../papers/wh0_arxiv_2606_22136.md)
- 仓库：[chenyt31_wh0.md](../repos/chenyt31_wh0.md)
