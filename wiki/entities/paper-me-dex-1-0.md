---
type: entity
tags:
  - paper
  - wam
  - tactile
  - manipulation
  - bimanual-manipulation
  - world-action-models
  - li-auto
  - open-source
status: complete
updated: 2026-11-22
arxiv: "2609.21449"
code: https://github.com/MachEmbodied/ME-Dex-1.0
original:
  - ../../sources/papers/me_dex_1_0_arxiv_2609_21449.md
  - ../../sources/repos/machembodied-me-dex-1-0.md
related:
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ./robotwin.md
  - ./paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md
  - ./paper-sa-2606-13877-contactworld-what-matters-in-vision-tactile-worl.md
  - ./paper-sa-2606-26663-tactile-wam-touch-aware-world-action-model-with.md
  - ./paper-sa-2603-23481-vtam-video-tactile-action-models-for-complex-phy.md
  - ./paper-me-brain-1-0.md
  - ./paper-me-u0.md
  - ../overview/li-auto-machembodied-4-papers-technology-map.md
sources:
  - ../../sources/papers/me_dex_1_0_arxiv_2609_21449.md
  - ../../sources/sites/me-dex-1-0.md
  - ../../sources/repos/machembodied-me-dex-1-0.md
  - ../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md
summary: "ME-Dex 1.0（arXiv:2609.21449，理想汽车 Foundation Model）：三专家 MoT WAM 联合预测未来视频/触觉/动作；H-Bridge 共享注意力 + Canonical Hand + Unified Tactile AE；RoboTwin avg 78.9% Clean→Random；推理 runtime 已开源。"
---

# ME-Dex 1.0：异构触觉进入 World Action Modeling

**ME-Dex 1.0**（*Bringing Heterogeneous Tactile Sensing into World Action Modeling*，[arXiv:2609.21449](https://arxiv.org/abs/2609.21449)，[项目页](https://machembodied.com/ME-Dex/ME-Dex1.0.html)，[代码](https://github.com/MachEmbodied/ME-Dex-1.0)）由 **理想汽车（Li Auto）Foundation Model** 提出：**MachEmbodied-Dex-1.0** 把触觉与视频并列建模为 **未来观测**，在 Mixture-of-Transformers 内用 flow matching 联合预测 future video、future tactile latents 与 action chunks。[2026-09-25 理想四篇盘点](../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md) 强调 **插接/夹持** 类任务需触觉未来，与 ME-Brain 真机插充电器低成功率形成互补读法（**复用本页**）。

## 一句话定义

**触觉不该只是条件输入——ME-Dex 1.0 把 future tactile 与 future video 一起预测，再让动作专家在联合去噪里读两种动力学。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来观测与动作生成的具身策略 |
| MoT | Mixture-of-Transformers | 视频/触觉/动作三专家并行 Transformer |
| AE | Autoencoder | Unified Tactile AE 编码异构触觉到共享潜空间 |
| SR | Success Rate | 闭环任务成功率 |
| HF | Hugging Face | 权重托管 |

## 为什么重要

- **范式推进：** 相对「VLA/WAM + 当前触觉条件」（Figure 1a–b），ME-Dex 1.0 把 **future tactile** 与 **future video** 并列作为预测目标（Figure 1c），动作生成可读取两种未来动力学表示。
- **跨具身触觉：** Canonical Hand Model + Unified Tactile AE 解决 **Representation Gap**——多手型/夹爪/传感布局映射到共享空间后再联合训练。
- **数据引擎：** Agentic Tactile Data Engine 在 RoboTwin / DexJoCo 轨迹回放时从仿真力传感器录触觉，缓解 **Data Gap**（原生 benchmark 无触觉）。
- **工程触点：** RoboTwin Clean→Random **78.9% avg**（论文 Table 1 口径），2026-10 开源 VAE 变体（`main@a2db63b`）更达 **82.5% avg**；**推理 runtime、Clean50 触觉训练脚本与 HF 数据集/权重现均已开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 理想汽车（Li Auto）Foundation Model |
| **骨干初始化** | Video / Action Expert 自 [Motus](./paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md) 预训练权重；Tactile Expert 随机初始化 |
| **训练目标** | flow matching 联合去噪：future video + future tactile latents + action chunks |
| **触觉对齐** | Canonical Hand Model（人手启发模板）+ Unified Tactile AE/VAE（跨源预训练后 **冻结**） |
| **数据** | RoboTwin 50 任务 27500 轨迹（Clean+Random）；DexJoCo 11 任务 1100 demo；ManiFeel 4 任务各 50 demo |
| **开源** | **已开源（Clean50 训练 + 推理）**：[MachEmbodied/ME-Dex-1.0](https://github.com/MachEmbodied/ME-Dex-1.0)（`runtime/` + `training/`）；HF 权重（AE 80.4% / VAE 82.5%）与 [Clean50 触觉数据集](https://huggingface.co/datasets/liuxuetao/MachEmbodied-Dex1.0-RoboTwin-Clean50-Tactile) 均已发布 |

## 核心原理

ME-Dex 1.0 围绕三个核心缺口展开统一建模：通过 **Canonical Hand Model** 与 **Unified Tactile AE/VAE** 对齐异构传感器与多手型空间表示（Representation Gap）；以三专家 **MoT + H-Bridge 共享注意力** 将未来触觉与未来视频并列作为 flow matching 联合去噪目标（Model Gap）；并以 **Agentic Tactile Data Engine** 在仿真回放中录制力传感器信号补齐触觉监督（Data Gap）。

### 流程总览

```mermaid
flowchart TB
  subgraph repr [统一触觉表示]
    chm[Canonical Hand Model]
    uae[Unified Tactile AE\n冻结]
  end
  subgraph mot [三专家 MoT + H-Bridge]
    vid[Video Expert\nfuture video]
    tac[Tactile Expert\nfuture tactile latents]
    act[Action Expert\naction chunks]
  end
  subgraph data [Agentic Tactile Data Engine]
    rt[RoboTwin 回放录力]
    dj[DexJoCo 回放录力]
  end
  obs[RGB + 当前触觉 + 指令] --> chm
  chm --> uae
  uae --> mot
  obs --> vid
  vid <-->|H-Bridge shared attention| tac
  tac <-->|H-Bridge shared attention| act
  data --> uae
  act --> robot[闭环操控]
```

## 评测

| 平台 | 设定 | 主要结果 |
|------|------|----------|
| **RoboTwin** | Clean+Random 混合训练 | Clean **91.56%** / Random **91.92%**（Table 1 Full Joint + H-Bridge） |
| **RoboTwin** | 仅 2500 Clean 训练；评测时触觉置零 | Clean **89.6%** / Random **68.1%**，**avg 78.9%**（超 OLA-Sem +6.5 pt avg） |
| **DexJoCo** | 11 任务 multi-task，rand-obj | **65.3% avg**；双手 avg **48.8%**（超 DECO/DECO.p 39.6%） |
| **ManiFeel** | 4 插入任务，单任务 50 demo | **70.0% avg**（+15 pt vs Diffusion Policy baseline） |
| **真机** | LeRobot SO-101 + Paxini PX6AX；Xynova Flex2 双手 | 定性：换插座、叠罐等 contact-rich 序列 |

**消融读法（RoboTwin Random）：** 仅 VA **86.90%** → +当前触觉条件 **89.54%** → +future tactile 预测 **90.60%** → +H-Bridge **91.92%**。训练后评测时将当前触觉置零仍 **91.70%**，说明 future tactile 预测路径可部分补偿缺失当前触觉。

## 与其他工作对比

同为「把触觉接进操作策略」，差别在 **触觉出现在计算图的哪一侧**：

| 维度 | ME-Dex 1.0（本页） | [ContactWorld](./paper-sa-2606-13877-contactworld-what-matters-in-vision-tactile-worl.md) / [Tactile-WAM](./paper-sa-2606-26663-tactile-wam-touch-aware-world-action-model-with.md) | 把触觉只当条件输入的 VLA |
|------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| 触觉的角色 | **与视频并列的未来观测**，被联合预测 | 视触觉世界模型，各自定义触觉进入方式 | 仅作为观测条件，不被预测 |
| 结构 | 三专家 MoT + **H-Bridge 共享注意力** | 见各自页 | 单骨干 + 动作头 |
| 跨传感器对齐 | Canonical Hand Model + **冻结** Unified Tactile AE | 依赖各自传感器口径 | 通常绑定单一传感器 |
| 代价 | 需要成规模的配对触觉数据与回放录力管线 | 同类代价 | 最低 |

- **「预测触觉」比「读触觉」多要求什么：** 联合去噪要求触觉信号 **可预测且与动作因果相关**；传感器噪声大或标定漂移时，这一支会变成噪声源而不是信息源——这也是 Unified Tactile AE 需要跨源预训练后 **冻结** 的原因。
- **数值可比性：** RoboTwin avg **78.9%（Clean→Random）** 是仿真基准口径，属 [评测闭环](../queries/embodied-eval-benchmark-selection-loop.md) 的 ③ 层；与 DexJoCo / ManiFeel 上的数不是同一评测面，也不蕴含真机成功率。
- **复现边界：** 仓库当前已同步开放 **RoboTwin Clean50 分布式训练（`training/`）+ Clean50 触觉数据集 + XPolicyLab 推理评测（`runtime/`）**；但 DexJoCo / ManiFeel / 真机数据与完整 Agentic Data Engine 自动化编排尚未全部释放，对比时仍需注明范围。

## 结论

**ME-Dex 1.0 把「触觉条件 WAM」推进到「未来触觉–视频–动作三模态联合 WAM」，并用统一手模 + 数据引擎把异构传感与缺触觉 benchmark 接进同一训练栈。**

- future tactile 与 future video 并列预测，是比纯条件输入更完整的 contact-rich 世界建模
- H-Bridge 共享注意力在 Random 上带来约 +1.3 pt，三专家信息交换有实证增益
- Unified Tactile AE 跨 RoboTwin/DexJoCo/ManiFeel contact F1 90%+，支撑跨布局联合训练
- RoboTwin Clean→Random **78.9%**（论文）与开源 VAE 版 **82.5% avg**（88.7% Clean / 76.2% Random）验证了泛化优势
- 双手任务（Assembly/Microwave 等）相对 DECO 系有明显优势，但并非全任务 uniform SOTA
- **开源边界清晰：** 仓库现已包含 Clean50 训练脚本、`tactile_force_field.py` 回放及 XPolicyLab 评测入口；Leaderboard 评测仍采用当前触觉置零 + 预测 future tactile 协议

## 源码运行时序图

图依据：基于 `MachEmbodied/ME-Dex-1.0@a2db63b` 的 `README.md`、`runtime/policy.py`、`training/train.py`、`training/build_t5_cache.py` 与 `training/configs/clean50_uni.yaml` 源码核实。

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant HF as HF 权重/Clean50 数据 + Wan2.2
    participant Train as training.train (torchrun 2x16)
    participant XPL as XPolicyLab (policy/ME_Dex_1_0)
    participant Pol as MEDexPolicy (runtime/policy.py)
    Dev->>HF: 下载策略权重、Wan2.2 VAE/T5 与 Clean50 数据集
    opt Clean50 训练路径
        Dev->>Train: build_t5_cache.py 生成 t5_prompt_table.pt
        Dev->>Train: clean50_uni.yaml (full_joint/h_bridge) 或 clean50_vae.yaml
        Train-->>Dev: 输出联合视频-触觉-动作 checkpoint
    end
    Dev->>XPL: 复制 runtime/ 并通过 eval.sh 启动 RoboTwin 评测
    loop episodes
        XPL->>Pol: head/left/right RGB -> camera_mosaic(BGR, 无归一化)
        Note over Pol: 当前触觉置零 + support_mask；联合去噪预测未来触觉与动作
        Pol-->>XPL: action chunk
    end
    XPL-->>Dev: Clean / Random SR
```

## 工程实践

| 主题 | 说明 |
|------|------|
| 视觉契约 | `camera_mosaic` 拼接头+双腕后按 `input_color_order: bgr` 转换，不做 mean/std 归一化（与 checkpoint 对齐） |
| 无触觉评测 | Leaderboard 无 force 观测：`MEDexPolicy` 将当前触觉置零并保留 `support_mask`，靠模型预测 future tactile |
| 权重与缓存 | 推理需 `Wan2.2-TI2V-5B` 的 VAE/T5/tokenizer；训练前须先运行 `python -m training.build_t5_cache` 生成 `t5_prompt_table.pt` |
| 训练配置 | `clean50_uni.yaml`（AE，2 观测帧 + 16 未来帧）与 `clean50_vae.yaml`（VAE，1 观测帧 + 16 未来帧），支持 `topology: full_joint` / `h_bridge` |
| 对照基线 | Fast-WAM / π₀.₅ / Motus；+Tac 变体用同一 Unified Tactile AE 编码 |

## 局限与风险

- **统一触觉编码器规模与传感器覆盖受限（论文 §5）：** 当前预训练受限于触觉数据规模与传感布局多样性，跨全新未见传感布局的系统性泛化仍待扩大验证。
- **chunk 间开环与高频局部力控缺口（论文 §5）：** 当前策略主要在 action chunk 间通过观测更新与重规划利用触觉，尚未结合块内高频触觉局部反馈控制来实时微调精细接触。
- **数据引擎开源范围边界：** 仓库已开源 Clean50 训练与 `robotwin_tactile` 回放代码，但完整 DexJoCo / ManiFeel 与真机数据集未随本仓发布。
- **项目页不可达与任务方差：** `machembodied.com` SSL 握手不稳定；ManiFeel Gear Assembly 略低于 baseline，DexJoCo 部分单臂任务 DECO 仍更强。

## 关联页面

- [双臂操作](../tasks/bimanual-manipulation.md) — 对照双手协调任务中视觉、触觉与动作接口的要求。
- [World Action Models（WAM）](../concepts/world-action-models.md)
- [RoboTwin](./robotwin.md)
- [Motus（索引）](./paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)
- [ContactWorld](./paper-sa-2606-13877-contactworld-what-matters-in-vision-tactile-worl.md)
- [Tactile-WAM（TAAM）](./paper-sa-2606-26663-tactile-wam-touch-aware-world-action-model-with.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [me_dex_1_0_arxiv_2609_21449.md](../../sources/papers/me_dex_1_0_arxiv_2609_21449.md)
- [me-dex-1-0.md](../../sources/sites/me-dex-1-0.md)
- [machembodied-me-dex-1-0.md](../../sources/repos/machembodied-me-dex-1-0.md)
- [wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md](../../sources/blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)

## 推荐继续阅读

- [arXiv:2609.21449](https://arxiv.org/abs/2609.21449)
- [GitHub 推理仓库](https://github.com/MachEmbodied/ME-Dex-1.0)
- [HF RoboTwin Clean2Random 权重](https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard)
- [Motus 项目页](https://motus-robotics.github.io/motus)
