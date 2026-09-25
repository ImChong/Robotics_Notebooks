# ME-VLM: A Unified VLM for Embodied Cognition and Agent Coordination

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24526>
- **项目页：** <https://machembodied.com/ME-Brain/ME-VLM.html>
- **代码：** <https://github.com/MachEmbodied/ME-VLM>（技术报告已释；推理/训练/权重 TODO）
- **入库日期：** 2026-09-25
- **索引来源：** [具身智能研究室 四篇盘点](../blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md)
- **一句话说明：** 统一 VLM（4B / 35B-A3B）融合具身认知与数字 Agent；双专家 RL + 多教师 on-policy 蒸馏；ME-Brain 认知核训练路线。

## 核心摘录

1. 训练管线：**具身能力注入**（两阶段 SFT）→ **具身专家 / Agent 专家分路 RL** → **多教师蒸馏** 合并为单一部署模型。
2. ME-VLM 35B-A3B 报告具身基准平均 **70.9**、Agent 基准平均 **72.5**（多项基准综合分；真机大样本成功率文内未单列）。
3. 与 ME-Brain 关系：ME-Brain 描述认知核如何与记忆、动作协同；ME-VLM 描述该认知核如何训练；**同一系统背景下的案例不宜重复计为两套独立真机验证**。
4. 仓库 TODO（2026-09-22）：技术报告 **已发布**；推理代码、训练代码、4B/35B 权重、端侧部署工具包 **待发布**。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-me-vlm.md`](../../wiki/entities/paper-me-vlm.md)
- 技术地图：[`wiki/overview/li-auto-machembodied-4-papers-technology-map.md`](../../wiki/overview/li-auto-machembodied-4-papers-technology-map.md)
