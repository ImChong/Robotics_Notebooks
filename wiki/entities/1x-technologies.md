---
type: entity
tags: [hardware, humanoid, industry, teleoperation, 1x-technologies]
institutions:
  - 1x-technologies
status: complete
updated: 2026-11-21
original:
  - ../../sources/repos/1x-technologies.md
  - ../../sources/papers/1x_world_model_redwood.md
  - ../../sources/repos/1xgpt.md
related:
  - ../overview/wam-motion-control-five-paths.md
  - ./paper-1xwm-redwood-world-model.md
  - ./humanoid-robot.md
  - ./figure-ai.md
  - ../queries/humanoid-hardware-selection.md
  - ../queries/humanoid-robot-data-collection-landscape.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/sites/company-roadmap-date-audit-2026-10-08.md
  - ../../sources/blogs/wechat_embodied_ai_lab_wam_motion_control_five_paths.md
  - ../../sources/blogs/leoinai_humanoid_robot_datacollection_2026-09-06.md
  - ../../sources/papers/1x_world_model_redwood.md
  - ../../sources/repos/1x-technologies.md
  - ../../sources/repos/1xgpt.md
summary: "1X Technologies（前身为 Halodi Robotics）是一家挪威与美国双总部的通用人形机器人公司，产品覆盖轮式人形 EVE 与面向家庭场景的双足 NEO，强调真实世界部署数据与 AI 能力迭代。"
---

# 1X Technologies

## 一句话定义

**1X Technologies** 专注于「能在真实环境里长期运行的人形机器人」，当前公开产品线以 **轮式人形 EVE**（面向仓储 / 安防 / 医疗等结构化场景）与 **双足 NEO**（强调家庭与通用场景）为主轴。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 人工智能 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| SDK | Software Development Kit | 软件开发工具包 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

- **硬件形态分叉**：同时押注「轮式 + 上半身人形」与「双足全身人形」，反映了商业化上对续航、通过性与成本的权衡。
- **数据闭环叙事**：家庭场景产品常与大规模真实交互数据、模仿学习与遥操作数据采集联系在一起（具体能力与时间表以官方为准）。LeoInAI Substack（2026-09）转述其使用 **Meta Quest 3** 让操作员完成困难任务以供机器人复现。
- **产业节点**：与美国人形赛道（Figure、Tesla、Agility 等）并行，代表欧洲与美国西海岸混合供应链路线的一支独立力量。

## 核心原理

1X Technologies 将硬件安全合规性（肌腱驱动 + 软体晶格包覆）与数据闭环 AI（机载 VLA 策略 Redwood + 离线世界模型评测 1XWM / 1xgpt）耦合在同一垂直产品体系中。

```mermaid
flowchart TB
  HW["EVE (轮式) / NEO (双足, 22-DoF 手, Revo1 肌腱驱动)"] --> Teleop["Meta Quest 3 / 1X Expert 遥操作与家用采集"]
  Teleop --> Data["多任务成功与失败轨迹数据"]
  Data --> Policy["Redwood AI 策略 (160M VLA, ~5 Hz 机载 Jetson Thor)"]
  Data --> WM["1XWM 世界模型 (反事实未来帧 + State-Value 离线筛 checkpoint)"]
  WM --> Policy
  Policy --> HW
  Data -. "EVE 第一人称 VQ-Token 子集 (Apache-2.0)" .-> Challenge["1xgpt World Model Challenge (GENIE 基线)"]
```

- **硬件代际与平台演进**：2014 年作为 Halodi Robotics 创立，2018 年推出 Revo1 低减速比肌腱驱动伺服，2022 年将轮式 EVE 部署于工业现场，2024–2025 年依次推出家用双足 **NEO Beta / Gamma** 并规划 2026 年 Early Access 交付。
- **软件与评测双轮**：前线控部署依靠约 160M 参数的机载策略 [Redwood](./1x-redwood-policy.md) 联合输出手臂、步态与骨盆姿态；研发迭代则依托 [1X World Model](./paper-1xwm-redwood-world-model.md) 对海量候选 checkpoint 做离线动作可控评测与排序。

## 工程实践

- **NEO 公开工程参数（官网规格）**：自重约 30 kg（66 lbs）、身高 5'6"、全身 75 DoF（双手各 22 DoF、双臂各 7 DoF、颈 3、脊柱 2、双腿各 6），842 Wh 电池标称续航 4 小时，声噪 22 dB，算力采用 1X NEO Cortex（NVIDIA Jetson Thor，标称最高 2070 FP4 TFLOPS）与双目 8.85 MP 90 Hz 鱼眼相机。
- **人机协作运营闭环**：常规家务由机载 Redwood AI 自主运行；遇到未掌握任务时通过「1X Expert」预约远程示教完成作业并反哺训练数据。
- **可运行开源边界**：商业产品栈（NEO 固件、Redwood 策略权重、完整 1XWM 生产评测）均未开源；社区可直接运行的是 `1x-technologies/1xgpt`（World Model Challenge 的 EVE 第一人称 token 数据与 GENIE 风格时空 Transformer 基线）。

## 产品与路线（归纳）

| 系列 | 典型定位 | 备注 |
|------|-----------|------|
| **EVE** | 机构 / 工业场景的轮式人形 | 强调可部署性与导航–操作组合 |
| **NEO** | 家庭与通用场景的双足平台 | Beta / Gamma 等代际迭代见官网；面向消费者的交付与条款变化较快 |

## 常见误区或局限

- **不要把融资报道当规格书**：估值与融资轮次与整机参数无直接对应关系。
- **学术可得性**：与 Unitree 等科研平台相比，1X 更偏产品与试点部署，论文级 URDF / SDK 开放度需单独核实。
- **自主 vs 遥操作**：公开演示中常见 VR 遥操作与高监督采集；「完全自主居家」仍是整个行业未解决的问题。

## 公司路线中的版本定位

- [Redwood 控制策略（与 World Model 区分）](./1x-redwood-policy.md)：与本页互补，按独立发布物核对架构和开放范围。

## 源码运行时序图

本图对应公司路线中与 1X World Model Challenge 配套的开源基线仓库 `1x-technologies/1xgpt`（GENIE 风格时空 Transformer + MaskGIT 采样），不覆盖 NEO/Redwood 等闭源产品栈。

入口：`./build.sh` 建环境并下载数据 → `python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir data/genie_model` → `python genie/generate.py --checkpoint_dir <ckpt>` → `python visualize.py --token_dir <dir>`

```mermaid
sequenceDiagram
    participant U as 用户
    participant T as train.py main()
    participant D as data.py RawTokenDataset + get_maskgit_collator
    participant M as genie/st_mask_git.py STMaskGIT
    participant G as genie/generate.py main()
    participant V as visualize.py main() + decode_latents_wrapper
    participant Q as magvit2 VQModel（data/magvit2.ckpt）
    U->>T: python train.py --genie_config ...
    T->>D: RawTokenDataset(train/val_data_dir, window_size, stride)
    T->>M: GenieConfig.from_pretrained 后构造 STMaskGIT(config)
    loop 每个 batch
        T->>D: DataLoader 经 collate_fn 做 token 加噪与按 cosine_schedule 掩码
        T->>M: model(input_ids, labels) 即 forward 调用 compute_logits
        M-->>T: 仅在掩码位置上的因子化交叉熵 loss 与 acc
        T->>T: accelerator.backward 后 optimizer.step
    end
    T-->>U: save_checkpoint 写出 final_checkpt
    U->>G: python genie/generate.py --checkpoint_dir ...
    G->>M: STMaskGIT.from_pretrained 取 val 样本窗口，后续帧置 mask_token_id
    loop timestep 从 num_prompt_frames 到 window_size-1
        G->>M: maskgit_generate(prompt_THW, out_t, maskgit_steps, temperature)
        M-->>G: 该帧 token 样本
        G->>G: 自回归写回 prompt_THW[:, timestep]
    end
    G-->>U: 写出 video.bin 与 metadata.json
    U->>V: python visualize.py --token_dir ...
    V->>Q: VQModel 加载 tokenizer 并解码 token
    Q-->>V: 图像帧
    V-->>U: 保存 GIF 与漫画条 PNG
```

图依据：1x-technologies/1xgpt@0069734，读过 `README.md`、`build.sh`、`train.py`（主循环与数据加载段）、`data.py`、`genie/st_mask_git.py`、`genie/generate.py`、`genie/evaluate.py`（`GenieEvaluator` 前段）、`visualize.py`。`train.py` 中 Llama 分支为注释/可选，未画。

## 关联页面

- [1X World Model（1XWM）](./paper-1xwm-redwood-world-model.md)
- [Robbyant（蚂蚁灵波）](./robbyant.md) — LingBot-World 动作条件视频世界模型对照
- [WAM×运动控制五路径](../overview/wam-motion-control-five-paths.md)
- [人形机器人](./humanoid-robot.md)
- [Figure AI](./figure-ai.md)（美国人形与 VLA 路线的可比节点）
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)
- [人形数据采集产业地图](../queries/humanoid-robot-data-collection-landscape.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)

## 公司路线日期口径

1X 官方公告：2024-08-30 发布 NEO Beta；是家庭双足原型事件，不是 EVE 首发或 NEO 商业交付日。 [日期证据](https://www.1x.tech/discover/announcement-1x-unveils-neo-beta-a-humanoid-robot-for-the-home)。详见[本轮日期核查](../../sources/sites/company-roadmap-date-audit-2026-10-08.md)；版本事件与原始产品首发分别记录。

## 参考来源

- [公司路线日期核查](../../sources/sites/company-roadmap-date-audit-2026-10-08.md)

- [1X Technologies 原始资料](../../sources/repos/1x-technologies.md)
- [1XWM 技术报告归档](../../sources/papers/1x_world_model_redwood.md)
- [1xgpt Challenge 归档](../../sources/repos/1xgpt.md)
- 源码核对（2026-11-02）：1x-technologies/1xgpt@0069734，读过 README.md、build.sh、train.py、data.py、genie/st_mask_git.py、genie/generate.py、genie/evaluate.py、visualize.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [1X 官网 About](https://1x.tech/about)
- [IEEE Robots — EVE](https://robots.ieee.org/robots/eve)
