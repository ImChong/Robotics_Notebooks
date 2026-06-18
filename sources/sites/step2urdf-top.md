# step2urdf.top

> 来源归档

- **标题：** step2urdf — STEP 转 URDF 在线工具
- **类型：** site（Web 应用部署）
- **来源：** Democratizing-Dexterous / step2urdf 项目
- **链接：** https://step2urdf.top/
- **入库日期：** 2026-06-18
- **一句话说明：** step2urdf 的**官方在线实例**：免费浏览器内 STEP→URDF，面向 ROS/ROS2 机器人模型快速生成；页面强调 **无需安装、一键导出**，底层与开源仓同源（OpenCascade.js + Vue）。
- **沉淀到 wiki：** [step2urdf](../../wiki/entities/step2urdf.md)

---

## 站点要点

### 定位（meta / 首页）

- **中文标题：** step2urdf — STEP 转 URDF 在线工具
- **描述：** 免费在线 STEP 转 URDF，支持 ROS/ROS2 机器人模型生成，无需安装
- **关键词：** STEP转URDF、ROS 模型、机器人仿真、URDF 生成、opencascade

### 与源码仓关系

- 前端为 Vite 构建的 SPA（`#app` 挂载）；功能与 [step2urdf GitHub 仓](../repos/step2urdf.md) README 一致。
- HTML 中预连接 `urdf.d-robotics.cc`（可能与 URDF 预览或关联服务有关）；**STEP 解析仍应在浏览器本地完成**（见仓库 Privacy 说明）。

### 推荐使用场景

| 场景 | 说明 |
|------|------|
| **快速验证** | 不想本地搭 pnpm 环境，直接用在线版试 STEP→URDF |
| **教学/demo** | 向团队展示 CAD 到仿真模型的最短路径 |
| **敏感模型** | 若担心第三方托管，应 **自托管 Release 包** 或本地 `pnpm dev` |

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [step2urdf.md](../repos/step2urdf.md) | 源码、特性与技术栈 |
| [urdf-studio.md](../repos/urdf-studio.md) | 更完整的 URDF/MJCF **设计与 BOM** 工作站 |
| [robot-viewer.md](../repos/robot-viewer.md) | 导出 URDF 后的查看与轻量仿真 |
