# 技术栈项目执行清单 v21

最后更新：2026-04-21（V21 启动，基于 V20 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v20.md`](tech-stack-next-phase-checklist-v20.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V20 交付基线 (V21 起点)

| 维度 | V20 状态 | V21 目标 |
|------|-----------|---------|
| 知识图谱节点 | 217 | **≥ 225** |
| 知识图谱边数 | 1255 | **≥ 1400** |
| 事实库 (CANONICAL_FACTS) | 115 条 | **≥ 140 条** |
| 交互体验 | 解释预览 + 磁吸模式 | **实现详情页面的“知识地图”迷你浮窗** |
| 技术专题 | Scaling Law | **建立“触觉与力觉闭环（Haptics）”专题** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **语义搜索“智能拼写纠错”**：
    - [x] `scripts/search_wiki.py` 集成基础的编辑距离算法，当查询无结果时自动推荐最接近的 Tag 或 标题。
- [x] **自动化背链一致性 Lint**：
    - [x] `scripts/lint_wiki.py` 新增检测：`formalizations/` 中的公式变量（如 $J, M, q$）在正文描述中必须有对应的物理含义解释。
- [x] **图谱导出数据精简**：
    - [x] 优化 `scripts/generate_link_graph.py`，移除节点冗余 `community_label` 字段（前端从 `communities` 数组查表），`exports/link-graph.json` 体积从 168 KB 降至 159 KB。

## P1: 触觉与力觉闭环 (Haptics) 专题 (Quality)

- [ ] **建立触觉学习知识链 (+3)**：
    - [x] `wiki/concepts/visuo-tactile-fusion.md` (视触觉融合：在接触瞬间如何平衡视觉全局与触觉局部信息)。
    - [ ] `wiki/methods/tactile-impedance-control.md` (基于触觉反馈的阻抗控制：实现自适应力度的灵巧抓取)。
    - [x] `wiki/formalizations/contact-wrench-cone.md` (接触力旋量锥：处理多点接触与力矩平衡的数学形式化)。
- [ ] **深化灵巧手感知**：
    - [ ] `wiki/entities/gel-slim.md` (下一代超薄视觉触觉传感器：硬件特性与仿真建模)。

## P2: 硬件通信与底层实时链路形式化 (Quantity)

- [ ] **发布通信层形式化定义 (+3)**:
    - [ ] `wiki/formalizations/control-loop-latency-modeling.md` (控制环路延迟建模：总线、计算与内核调度延迟的数学求和)。
    - [ ] `wiki/formalizations/udp-multicast-dynamics.md` (UDP 组播动力学：在 LCM 等中间件中的数据包丢失与一致性形式化)。
    - [ ] `wiki/concepts/clock-synchronization-algorithms.md` (时钟同步算法：PTP/DC 协议在多板卡运控中的实现原理)。
- [ ] **新增对比页 (+1)**:
    - [ ] `wiki/comparisons/ethercat-vs-ethernet-ip.md` (工业总线对比：确定性、拓扑能力与机器人实时控制适配度)。

## P3: 交互层“情境化”导航优化 (UX/UI)

- [ ] **详情页“当前位置”微地图**：
    - [ ] 修改 `docs/main.js`，在页面顶部显示一个微小的 D3 局部图谱，展示当前节点及其 1-hop 邻居，实现“一叶知秋”的导航体验。
- [ ] **搜索结果按“置信度”分级**：
    - [ ] 优化 UI，将搜索结果分为“精确匹配”与“潜在关联”两个区块。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 190**。
- [ ] 事实库扩展至 **140 条** 以上。
- [ ] 详情页“微地图”组件上线并正常工作。
- [ ] `log.md` 记录 V21 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
