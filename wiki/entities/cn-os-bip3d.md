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
  - ../../sources/repos/bip3d.md
summary: "地平线 开源项目 BIP3D（工程与工具）：感知/定位/建图模块：为导航与控制提供环境状态…"
institutions:
  - horizon-robotics
---

# BIP3D

## 一句话定义

**BIP3D** 是 [地平线](https://github.com/HorizonRobotics) 公开的 **工程与工具** 开源项目：感知/定位/建图模块：为导航与控制提供环境状态

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
| 机构 | 地平线 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/HorizonRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `BIP3D` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`bash engine.sh configs/bip3d_det.py`（单卡 → `python ./tools/train.py <config> --work-dir ./work-dir`，多卡 → `tools/dist_train.sh`）；评测 `bash test.sh <config> <ckpt>`（单卡 → `python3 tools/test.py`）。

```mermaid
sequenceDiagram
    participant U as 用户 / engine.sh、test.sh
    participant T as tools/train.py、tools/test.py
    participant R as mmengine Runner
    participant D as bip3d/datasets<br/>EmbodiedScanDetGroundingDataset + transforms
    participant P as bip3d/models/data_preprocessors<br/>CustomDet3DDataPreprocessor
    participant M as bip3d/models/structure.py<br/>BIP3D(BaseDetector)
    participant F as feature_enhancer.py /<br/>spatial_enhancer.py
    participant DEC as bip3d/models/bbox3d_decoder.py<br/>BBox3DDecoder (+ instance_bank.py)
    participant E as bip3d/eval<br/>metrics（IndoorDetMetric）

    U->>T: engine.sh → python tools/train.py configs/*.py
    T->>T: from bip3d import *（注册模型/数据集）；Config.fromfile
    T->>R: Runner.from_cfg(cfg)；runner.train()
    loop IterBasedTrainLoop（max_iters）
        R->>D: 取 batch（pipeline：多视角图像/深度加载、增强、文本）
        R->>P: 预处理 / 打包
        R->>M: model.loss(batch_inputs, batch_data_samples)
        M->>M: _forward → extract_feat（backbone[Swin] + neck；可选 backbone_3d 处理深度）
        M->>M: extract_text_feature（BertModel + text_feat_map）
        M->>F: feature_enhancer(feature_maps, feature_3d, text_dict)
        M->>F: spatial_enhancer(...) → depth_prob, loss_depth（若配置）
        M->>DEC: decoder(feature_maps, text_dict, depth_prob, ...)
        DEC->>DEC: instance_bank.get → anchor / instance_feature
        loop operation_order（默认 gnn、text_cross_attn、deformable、ffn、refine 循环）
            DEC->>DEC: 各层更新 instance_feature 与 anchor，输出 prediction / classification
        end
        M->>DEC: decoder.loss(model_outs, batch_inputs, text_dict)
        DEC-->>R: loss dict → OptimWrapper(AdamW) 反传
    end
    R-->>U: work-dir 下 checkpoint（CheckpointHook）

    U->>T: test.sh → python3 tools/test.py config ckpt
    T->>R: Runner.from_cfg；runner.test()
    R->>M: model.predict → decoder.post_process
    R->>E: test_evaluator = IndoorDetMetric（configs/bip3d_det.py）
    E-->>U: 评测结果
```

图依据：HorizonRobotics/BIP3D@b1ed6a8，读过 README.md、docs/quick_start.md、engine.sh、test.sh、tools/train.py、configs/bip3d_det.py（model 与 dataset 段落）、bip3d/models/structure.py、bip3d/models/bbox3d_decoder.py（forward 与 loss 的主干部分）、bip3d/models/instance_bank.py（方法清单）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [BIP3D 源码归档](../../sources/repos/bip3d.md)（<https://github.com/HorizonRobotics/BIP3D>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HorizonRobotics/BIP3D@b1ed6a8，读过 README.md、docs/quick_start.md、engine.sh、test.sh、tools/train.py、configs/bip3d_det.py（局部）、bip3d/models/structure.py、bip3d/models/bbox3d_decoder.py（局部）、bip3d/models/instance_bank.py（方法清单）；feature_enhancer / spatial_enhancer / 数据集 transforms / eval 模块内部未读；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [地平线 官方组织](https://github.com/HorizonRobotics)
