# 技术栈项目执行清单 v17

最后更新：2026-04-21（V17 启动，基于 V16 核心知识交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v16.md`](tech-stack-next-phase-checklist-v16.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V16 交付基线 (V17 起点)

| 维度 | V16 状态 | V17 目标 |
|------|-----------|---------|
| 知识图谱节点 | 131 | **≥ 150** |
| 知识图谱边数 | 776 | **≥ 950** |
| 事实库 (CANONICAL_FACTS) | 85 条 | **≥ 100 条** |
| VLA 覆盖 | 形式化定义已补齐 | **建立 VLA 训练数据流（Data Pipelines）知识链** |
| 交互体验 | 基础表格与联想 | **实现社区高亮与阅读足迹** |

---

## P0: 自动化知识挖掘 (Engineering)

- [ ] **自动化事实发现脚本**：
    - [ ] 编写 `scripts/discover_facts.py`，利用正则表达式或启发式规则扫描所有 Wiki 页面中的“一句话定义”，自动推荐新的 `CANONICAL_FACTS`。
- [ ] **Lint 规则严谨化**：
    - [ ] `scripts/lint_wiki.py` 新增检测：`methods/` 页面必须包含 `## 主要方法路线` 或类似区块。
    - [ ] 强制检查 `queries/` 页面中提到的“Query 产物”是否在背链中被正确引用。

## P1: 具身数据流水线 (Data Pipelines) 专题 (Quality)

- [ ] **新增数据处理核心页 (+3)**：
    - [ ] `wiki/concepts/embodied-data-cleaning.md` (具身数据清洗：异常轨迹剔除、时间戳同步、重定向误差过滤)。
    - [ ] `wiki/methods/actuator-network.md` (执行器网络：如何在仿真中建模复杂的非线性电机特性以减小 Sim2Real Gap)。
    - [ ] `wiki/formalizations/se3-representation.md` (SE(3) 位姿表示：欧拉角、四元数、旋转矩阵在深度学习中的优劣对比)。
- [ ] **扩充 VLA 部署经验**：
    - [ ] `wiki/queries/vla-deployment-guide.md` (扩充至 500 字，重点增加 TensorRT 加速与异步推理实现)。

## P2: 交互层“最后一百米” (UX/UI)

- [ ] **图谱社区焦点模式 (Community Focus)**：
    - [ ] 修改 `docs/graph.html`，支持点击图例（Legend）中的社区名称，一键高亮该社区所有节点并弱化背景。
- [ ] **沉浸式阅读足迹**：
    - [ ] 修改 `docs/main.js`，在详情页底部增加“最近访问”组件，基于 SessionStorage 展示最近阅读的 5 个页面。
- [ ] **对比页移动端横滑提示**：
    - [ ] 在 `docs/style.css` 中为 `.table-wrapper` 增加视觉提示（如右侧渐变遮罩），提示用户表格可横向滚动。

## P3: 外部实体与中间件补完 (Quantity)

- [ ] **新增中间件与遥控实体 (+2)**:
    - [ ] `wiki/entities/oculust-quest-teleop.md` (使用 VR 头显进行具身数据采集的硬件选型与配置)。
    - [ ] `wiki/entities/nvidia-omniverse.md` (Isaac Sim 的底层支撑平台，工业级具身模拟生态)。
- [ ] **新增对比页 (+1)**:
    - [ ] `wiki/comparisons/data-gloves-vs-vision-teleop.md` (灵巧操作数据采集：穿戴式手套 vs 视觉动捕的深度对比)。

---

## 验收标准 (Definition of Done)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 150**。
- [ ] 事实库扩展至 **100 条** 以上。
- [ ] 详情页“最近访问”功能上线并正常工作。
- [ ] `log.md` 记录 V17 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
