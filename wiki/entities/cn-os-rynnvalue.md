---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/rynnvalue.md
summary: "阿里巴巴 开源项目 RynnValue（具身Agent/规划）：模型对候选动作或执行轨迹进行价值判断，为策略选择、失败筛选和后训练提供反馈信号；它解决的是动作好坏的评估，不直接生成完整机器人控制命令。…"
institutions:
  - alibaba
---

# RynnValue

## 一句话定义

**RynnValue** 是 [阿里巴巴](https://github.com/amap-cvlab) 公开的 **具身Agent/规划** 开源项目：模型对候选动作或执行轨迹进行价值判断，为策略选择、失败筛选和后训练提供反馈信号；它解决的是动作好坏的评估，不直接生成完整机器人控制命令。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第四层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 阿里巴巴 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/amap-cvlab、https://github.com/alibaba-damo-academy |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RynnValue` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd rynn_infer && uv run python inference.py --model_path Alibaba-DAMO-Academy/RynnValue-8B --video_path ../example/Put_the_box_in_the_drawer_and_close_it.mp4 --instruction "..." --num_frames 8`（README Quickstart）

```mermaid
sequenceDiagram
    participant Main as rynn_infer/inference.py main()
    participant HF as transformers AutoConfig / AutoModel / AutoProcessor（trust_remote_code）
    participant Proc as rynn_value/processing_rynn_value_lang.py（process_episode）
    participant Model as rynn_value/modeling_rynn_value_lang.py RynnValueLangModel
    participant Plot as rynn_infer/plot_utils.py save_video_with_trend
    Main->>HF: AutoConfig.from_pretrained，强制 _attn_implementation = pred_slot_isolated_eager
    Main->>HF: AutoModel.from_pretrained 与 AutoProcessor.from_pretrained
    Main->>Main: load_video_frames（imageio 解码全部帧）、resize_frames
    Main->>Main: sample_frame_indices 得到前缀终点 eval_indices
    loop 每个前缀终点 end_idx（按 batch_size 成批）
        Main->>Proc: process_episode(instruction, 前缀内 num_frames 帧, robot/camera_description)
        Proc-->>Main: input_ids、pixel_values、image_grid_thw
        Main->>Model: model(**batch_kwargs)，inference_mode
        Model-->>Main: outputs.value.pred_value，取最后一个 slot 作为该前缀末帧剩余时间
    end
    Main->>Model: generate(最后一个前缀样本, max_new_tokens, do_sample=False)
    Model-->>Main: Analysis 文本
    Main->>Main: parse_analysis 正则提取 Video Description / Match / Success
    Main->>Plot: save_video_with_trend(images, pred_value, sampled_indices)
    Plot-->>Main: output_with_trend.mp4
```

图依据：alibaba-damo-academy/RynnValue@cd29294，读过 README.md、rynn_infer/inference.py、rynn_value/attention_impl.py（片段）、rynn_value/modeling_rynn_value_lang.py（类与头定义部分）；未读 processing_rynn_value_lang.py、plot_utils.py、robometer/、pi-rl/ 内部，故图仅覆盖 rynn_infer 推理入口。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RynnValue 源码归档](../../sources/repos/rynnvalue.md)（<https://github.com/alibaba-damo-academy/RynnValue>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：alibaba-damo-academy/RynnValue@cd29294，读过 README.md、rynn_infer/inference.py、rynn_value/attention_impl.py（部分）、rynn_value/modeling_rynn_value_lang.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [阿里巴巴 官方组织](https://github.com/amap-cvlab)
