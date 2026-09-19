# ShawnJoeng/Text2Mujoco

> 来源归档

- **标题：** Text2MuJoCo
- **类型：** repo（Agent Skill + MuJoCo 环境生成器）
- **维护方：** ShawnJoeng（GitHub 个人）
- **链接：** https://github.com/ShawnJoeng/Text2Mujoco
- **Showcase：** https://github.com/ShawnJoeng/Text2Mujoco/tree/main/showcase
- **许可证：** MIT
- **语言：** Python
- **入库日期：** 2026-09-19
- **一句话说明：** 在 Codex / Claude Code 等编码代理内运行的 **Agent Skill**：把自然语言场景描述编译为 **可加载、可验证** 的 MuJoCo 3 包（`scene_spec.json` + `model.xml` + `environment.py` + `interaction_manifest.json` + smoke tests），并以 JSON 报告与 RGB-D 证据链背书。
- **开源状态：** **已开源** — 完整仓库含 `text2mujoco_codex` / `text2mujoco_claude` 双适配器、showcase 八场景、校验脚本与 committed captures。
- **沉淀到 wiki：** [text2mujoco](../../wiki/entities/text2mujoco.md)

---

## 核心能力（摘录）

1. **一包式交付**：从单条 NL 请求生成一致的多文件包 — spec、MJCF、Python 环境 API、interaction manifest、两类 smoke test。
2. **统一交互 API**：`list_interaction_points()`、`get_action_schema()`、`reset()`、`step()`、`observe()`、`is_success()`；每个 affordance 为带 typed payload 与 dependency 的真实 handler。
3. **几何与物理诚实**：metric 尺寸 → MJCF half-size；`orientation_xyzw` → `quat="w x y z"`；resting body 按 half-size 就座；`conaffinity="7"` 防自碰撞穿透；抓取用 runtime `weld` 而非写死 `qpos`。
4. **分层验证**：spec/manifest 静态校验 → `physics_smoke.py`（`MUJOCO_GL=disable`）→ `render_smoke.py`（RGB-D）→ showcase 级 `model_audit.py` / `sequence_contact_test.py` / `dense_archive_test.py`。
5. **Showcase（8 场景）**：含 [01-button-cube-box](https://github.com/ShawnJoeng/Text2Mujoco/tree/main/showcase/01-button-cube-box)（4 interaction points，2.660 s sim）与 [04-lever-ball-ramp](https://github.com/ShawnJoeng/Text2Mujoco/tree/main/showcase/04-lever-ball-ramp)（杠杆开闸 + 球滚坡进托盘）。

## 验证报告（01-button-cube-box）

- **报告：** [TEST_REPORT.md](https://github.com/ShawnJoeng/Text2Mujoco/blob/main/showcase/01-button-cube-box/TEST_REPORT.md)（2026-09-09）
- **MuJoCo 版本：** 3.2.7
- **通过项：** skill 结构、scene spec validator（17 负例拒 / 4 正例收）、MJCF/manifest 静态、physics（actuator/gravity/5-collider box contact/reset/MJB reload）、RGB-D renderer（480×640×3 + depth）
- **关键物理断言：** 20 named objects；cube 0.2 kg；按钮 slide joint 到位；释放 cube 落点约 `[0.30, 0.18, 0.845]` m；序列 `press_start_button → grasp_red_cube → place_cube_in_box` 通过

## 安装入口（README）

- **Codex：** `install-skill-from-github.py --repo ShawnJoeng/Text2Mujoco --path text2mujoco_codex --name text2mujoco`
- **Claude Code：** 对称路径 `text2mujoco_claude`；CI diff 两适配器脚本防漂移

## 对 wiki 的映射

- 实体页：[text2mujoco](../../wiki/entities/text2mujoco.md)
- MuJoCo 生态：[mujoco](../../wiki/entities/mujoco.md)、[mujoco-playground](../../wiki/entities/mujoco-playground.md)
- 仿真选型：[simulator-selection-guide](../../wiki/queries/simulator-selection-guide.md)
- Agent Skill 对照：[archify](../../wiki/entities/archify.md)、[agent-skills-addyosmani](../../wiki/entities/agent-skills-addyosmani.md)
