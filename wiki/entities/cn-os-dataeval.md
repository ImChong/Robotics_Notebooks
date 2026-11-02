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
  - ../../sources/repos/dataeval.md
summary: "松灵机器人 开源项目 DataEval（数据集/Benchmark）：评测基准与工具…"
institutions:
  - agilex
---

# DataEval

## 一句话定义

**DataEval** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **数据集/Benchmark** 开源项目：评测基准与工具

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 松灵机器人 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `DataEval` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`bash process_data.sh <路径列表文件>`（逐行读取路径，依次调用下列三个 Python 脚本）

```mermaid
sequenceDiagram
    participant SH as process_data.sh
    participant M as mcap_visualization/mcap_to_aloha_data.py
    participant X as extract.py
    participant V as openvla/openvla-7b (HF AutoModelForVision2Seq)
    participant A as embedding_analysis.py

    loop 路径列表文件中的每一行 path
        SH->>M: python mcap_to_aloha_data.py --datasetDir data_sample/path --alohaYaml aloha_data_params.yaml
        M->>M: get_mcap_files() 然后 process_file()
        M->>M: rosbag2_py.SequentialReader 读 mcap，按 yaml 的话题配置落盘到 aloha/episode*/
        M-->>SH: 失败则 continue 跳过该路径
        SH->>X: python extract.py --data_root .../aloha --output_dir features/path/ --n_segments 3
        X->>V: init_model(): AutoProcessor / AutoModelForVision2Seq.from_pretrained
        loop 每个 episode 的 front / left / right 三个视角
            X->>V: processor(图像) 然后 vision_backbone.featurizer(pixel_values)
            V-->>X: 每帧 CLS 特征 vision_outputs[:, 0]
            X->>X: entropy_weighted_mean() 按 n_segments 分段聚合并拼接
        end
        X->>X: 三视角拼接，np.save(episodeN_fused.npy)
        SH->>A: python embedding_analysis.py --feature_dir features/path/ --output_dir features/path/
        A->>A: load_features() 然后 analyze_features()
        A->>A: 聚类、视角贡献度、时序一致性、TSNE、zscore 离群
        A-->>SH: 写 cluster_assignments.csv 与 advanced_visualization.png
    end
```

图依据：agilexrobotics/DataEval@2c8312c，读过 `README.md`、`process_data.sh`、`extract.py`、`embedding_analysis.py`（`load_features` / `analyze_features` / `main`，其中聚类细节仅看了函数名与 import）、`mcap_visualization/mcap_to_aloha_data.py`（`main` / `process_file` 开头）。仓库另含从 OpenVLA 拷入的 `prismatic/`、`experiments/`、`vla-scripts/`，本图未覆盖。

## 参考来源

- [DataEval 源码归档](../../sources/repos/dataeval.md)（<https://github.com/agilexrobotics/DataEval>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：agilexrobotics/DataEval@2c8312c，读过 README.md、process_data.sh、extract.py、embedding_analysis.py、mcap_visualization/mcap_to_aloha_data.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
