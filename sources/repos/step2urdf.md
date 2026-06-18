# step2urdf

> 来源归档

- **标题：** step2urdf（原 URDFlyS2U）
- **类型：** repo（Web CAD → URDF 转换工具）
- **来源：** Democratizing-Dexterous
- **链接：** https://github.com/Democratizing-Dexterous/step2urdf
- **在线版：** https://step2urdf.top/
- **Stars：** ~211（2026-06）
- **入库日期：** 2026-06-18
- **许可证：** MIT
- **一句话说明：** 浏览器端 **STEP → URDF** 转换工具：基于 **OpenCascade.js** 从 CAD 几何自动识别旋转/移动关节、估算惯量与质心，并提供交互式关节调试与层级树管理；**STEP 文件仅在本地处理，不上传服务器**。
- **沉淀到 wiki：** [step2urdf](../../wiki/entities/step2urdf.md)

---

## 核心定位（README）

- **仓库更名：** 官方名称由 `URDFlyS2U` 改为 `step2urdf`，后续引用应使用新名。
- **隐私与本地处理：** STEP 在**用户本机**完成解析与转换，不依赖后端上传；可用浏览器开发者工具（F12）验证网络无文件外传。
- **目标用户：** 机械 CAD 已产出 STEP 装配体、需要快速生成 **ROS/ROS2 可用 URDF** 的硬件/仿真工程师。

---

## 主要特性

| 能力 | 说明 |
|------|------|
| **几何驱动关节识别** | 从 STEP 中检测 **圆弧/直线段**，自动定义 **revolute** 与 **prismatic** 关节 |
| **惯量与质心** | 输入整机总质量（易测参数），自动分配各 link 惯量与 COM；可逐 link 微调 |
| **交互式关节可视化** | 导出前可滑动/测试各关节，确认轴向与限位 |
| **层级模型树** | 浏览导入 solid，选择、隐藏、组织组件 |
| **轴向微调** | XYZ 方向精细调整关节轴偏移，对齐 CAD 几何 |
| **关节类型** | 支持 **revolute** 与 **prismatic** |

---

## 技术栈（package.json）

- **前端：** Vue 3、Vite、TypeScript、Element Plus、Pinia、Three.js
- **CAD 内核：** `opencascade.js` 2.0.0-beta（WebAssembly，浏览器内解析 STEP）
- **并发：** Comlink（Worker 侧几何计算）
- **构建：** pnpm；本地开发 `pnpm dev`；Release 可用 `python -m http.server` 托管静态包

---

## 使用方式

1. **在线：** 打开 [step2urdf.top](https://step2urdf.top/)，加载本地 STEP，配置关节与质量，导出 URDF。
2. **自托管：** clone 后 `pnpm install && pnpm build`，或下载 GitHub Release 静态包本地起 HTTP 服务。

**视频教程：** [Bilibili BV168PjzrErB](https://www.bilibili.com/video/BV168PjzrErB)

---

## 与本库关系

| 资料 | 关系 |
|------|------|
| [step2urdf-top.md](../sites/step2urdf-top.md) | 官方在线部署站点归档 |
| [urdf-studio.md](urdf-studio.md) | **全流程 Web 设计工作站**（Skeleton/Detail/Hardware + 多格式导出）；step2urdf 专注 **CAD 导入 → URDF** 单点 |
| [cad-skills.md](../../wiki/entities/cad-skills.md) | Agent Skills：**build123d → STEP → URDF** CLI 路线；与本工具 **GUI + OpenCascade.js** 路线互补 |
| [robot-viewer.md](robot-viewer.md) | URDF 导出后的 **预览/轻量仿真** 下游工具 |

---

## 对 wiki 的映射

- 新建实体页 [`wiki/entities/step2urdf.md`](../../wiki/entities/step2urdf.md)：STEP→URDF 工作流、与 URDF-Studio/CAD Skills 选型对照、本地隐私处理说明。
- 更新 [`wiki/entities/urdf-studio.md`](../../wiki/entities/urdf-studio.md)、[`references/repos/utilities.md`](../../references/repos/utilities.md) 交叉引用。
