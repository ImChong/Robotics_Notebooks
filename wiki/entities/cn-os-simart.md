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
  - ../../sources/repos/simart.md
summary: "字节跳动机器人团队 开源项目 SimArt（仿真环境）：使用多模态模型把整体网格分解为可用于仿真的关节资产，服务机器人场景与交互资产构建。…"
institutions:
  - bytedance
---

# SimArt

## 一句话定义

**SimArt** 是 [字节跳动机器人团队](https://github.com/ByteDance-Seed) 公开的 **仿真环境** 开源项目：使用多模态模型把整体网格分解为可用于仿真的关节资产，服务机器人场景与交互资产构建。

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
| 机构 | 字节跳动机器人团队 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/ByteDance-Seed、https://github.com/bytedance |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `SimArt` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python inference/infer.py --object_path ./assets/box_00.glb --debug`（可选先 `python scripts/process_raw_objects.py --input <glb> --output ./assets --render` 做坐标对齐；权重默认放 `./checkpoints/simart_mllm` 与 `./checkpoints/simart_vqvae`）。

```mermaid
sequenceDiagram
    participant CLI as inference/infer.py<br/>main()
    participant LD as load_vqvae<br/>vqvae/sparse_structure_vqvae.py (VQVAE3D)
    participant P as load_raw_object_and_build_prompt
    participant R as utils/render_utils.py<br/>render_object (Blender)
    participant M as Qwen3VLForConditionalGeneration<br/>(simart_mllm)
    participant PS as utils/parse_utils.py
    participant U as utils/urdf_utils.py

    CLI->>CLI: collect_object_glbs(object_path)
    CLI->>M: from_pretrained(model_path)；AutoProcessor.from_pretrained
    CLI->>LD: load_vqvae(vqvae_ckpt_dir)
    loop 每个 GLB
        CLI->>P: 构建 QA 提示
        P->>P: trimesh 加载并按中心/最大边长归一化，导出 *_scaled.glb
        P->>R: render_object(单一视角, blender_path)
        P->>PS: file_to_mesh_words(glb, vqvae) → 校验 xyzK → sparse_mesh_to_voxel_sequence_xyzK
        P-->>CLI: messages = system_prompt + 渲染图 + 体素序列与问题
        CLI->>M: processor(...) → model.generate(max_new_tokens)
        M-->>CLI: JSON 响应 (parts_voxels, 各部件 type/center/axis 等)
        CLI->>PS: parse_parts_voxels_json / parse_parts_captions_json
        CLI->>LD: vqvae.Decode(部件体素 indices) 恢复部件点云（无有效 indices 则回退网格点）
        CLI->>U: segment_mesh_by_wrapped_pcd_no_minus1(mesh, 部件点云)
        CLI->>U: write_simple_urdf(部件 OBJ, captions)
        CLI->>CLI: 写 {name}.json（--debug 时另写 .ply）
    end
```

图依据：ByteDance-Seed/SimArt@c8d0829，读过 README.md、inference/infer.py（main、load_raw_object_and_build_prompt）；utils/、vqvae/、scripts/process_raw_objects.py 内部实现未读，仅按 infer.py 中的导入与调用关系标注。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [SimArt 源码归档](../../sources/repos/simart.md)（<https://github.com/ByteDance-Seed/SimArt>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：ByteDance-Seed/SimArt@c8d0829，读过 README.md、inference/infer.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [字节跳动机器人团队 官方组织](https://github.com/ByteDance-Seed)
