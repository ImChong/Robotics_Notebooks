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
  - ../../sources/repos/rynnec.md
summary: "阿里巴巴 开源项目 RynnEC（具身Agent/规划）：项目研究机器人怎样从多模态观测形成环境理解、任务分解和下一步决策，并把结果交给导航或操作模块执行；适合定位认知规划层与低层策略之间的接口。…"
institutions:
  - alibaba
---

# RynnEC

## 一句话定义

**RynnEC** 是 [阿里巴巴](https://github.com/amap-cvlab) 公开的 **具身Agent/规划** 开源项目：项目研究机器人怎样从多模态观测形成环境理解、任务分解和下一步决策，并把结果交给导航或操作模块执行；适合定位认知规划层与低层策略之间的接口。

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

1. 从官方 GitHub/Gitee 组织检索 `RynnEC` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python inference/gradio_demo.py --model-path Alibaba-DAMO-Academy/RynnEC-2B`（README 所述本地 Gradio 应用；另有 `inference/notebooks/` 三个 notebook）

```mermaid
sequenceDiagram
    participant U as 用户（Gradio 界面）
    participant D as inference/gradio_demo.py
    participant SAMhf as transformers SamModel（facebook/sam-vit-huge）
    participant I as rynnec/__init__.py model_init / mm_infer / mm_infer_segmentation
    participant M as rynnec/model/rynnec_qwen2.py RynnecQwen2ForCausalLM
    participant A as rynnec_arch.py prepare_inputs_labels_for_multimodal
    participant S as grounding_encoder（rynnec/model/sam2.py SAM2）
    D->>SAMhf: from_pretrained，用于把画笔点转成物体 mask
    D->>I: model_init(model_path)，返回 model 与 processor
    U->>D: 上传视频，load_video(fps=1, max_frames=128)
    U->>D: 在某帧涂抹，generate_masks_video
    D->>SAMhf: apply_sam(image, 点)，取 IOU 最高的 mask
    alt 模式为 QA
        U->>D: 提问，run_text_inference
        D->>I: mm_infer(帧与时间戳, instruction, masks, mask_ids)
        I->>M: generate(pixel_values, grid_sizes, masks, mask_ids)
        M->>A: 视觉编码，region token 位置由 region_encoder 嵌入物体 mask 特征
        A-->>M: inputs_embeds
        M-->>I: output_ids，decode 为文本
    else 模式为分割
        D->>I: mm_infer_segmentation(帧与时间戳, instruction)
        I->>I: DirectResize(1024)、sam_preprocess_batch 得到 sam_images
        I->>M: inference(...)，内部先 generate
        M->>S: 取 seg token 隐状态经 text_hidden_fcs，get_sam2_embeddings 与 language_embd_inference
        S-->>M: pred_masks
        M-->>I: 文本与 sigmoid 大于 0.5 的 mask
        I-->>D: colorize_masks 生成可视化
    end
    D-->>U: 文本回答 / mask 视频帧
```

图依据：alibaba-damo-academy/RynnEC@363c297，读过 README.md、inference/gradio_demo.py、rynnec/__init__.py、rynnec/model/rynnec_qwen2.py（generate、inference 部分）、rynnec/model/rynnec_arch.py（prepare_inputs_labels_for_multimodal 部分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RynnEC 源码归档](../../sources/repos/rynnec.md)（<https://github.com/alibaba-damo-academy/RynnEC>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：alibaba-damo-academy/RynnEC@363c297，读过 README.md、inference/gradio_demo.py、rynnec/__init__.py、rynnec/model/rynnec_qwen2.py（部分）、rynnec/model/rynnec_arch.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [阿里巴巴 官方组织](https://github.com/amap-cvlab)
