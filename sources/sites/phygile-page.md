# baojch.github.io/phygile-page（PhyGile 项目页）

> 来源归档（ingest）

- **标题：** PhyGile — Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking
- **类型：** site / project-page
- **官方入口：** <https://baojch.github.io/phygile-page/>
- **入库日期：** 2026-06-19
- **一句话说明：** 论文配套站点：三模块方法图（GMT MoE tracker / physics-prefix 生成微调 / TP-MoE 机器人原生扩散）、生成→微调→真机对比视频，以及 breakdance、侧手翻、旋跳等高动态真机演示。

## 页面公开信息（检索自 2026-06-19）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://baojch.github.io/phygile-page/> |
| arXiv | <https://arxiv.org/abs/2603.19305> |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **核心矛盾**：人体 text-to-motion + 重定向 → 几何合理但 **物理不可行**；需 **robot-native** 生成与 **可执行验证** 闭环。
2. **GMT（左）**：课程约束路由的两阶段 **MoE tracker** → 全局软后训练 + **动态专家扩展**；无标注运动后训练提升鲁棒性。
3. **生成（右）**：**TP-MoE** 条件 **扩散策略**，文本 → **262D** 机器人运动序列。
4. **微调（中）**：**可执行前缀 + 1s 新生成延续** → **预训练 GMT 验证** → **闭环仿真精炼** → physics-prefix 阶段 **微调 GMT**。
5. **结果叙事**：生成 / 微调 / 真机三段对比；真机含 breakdance、cartwheel、high kick、180°/360° spin jump 等。

## 对 wiki 的映射

- [`wiki/entities/paper-phygile.md`](../../wiki/entities/paper-phygile.md) — 方法栈、闭环管线与真机案例归纳
