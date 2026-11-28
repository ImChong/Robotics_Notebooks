# StackForce CAD2URDF（cad2urdf.stackforce.cc）

> 来源归档

- **标题：** StackForce CAD Importer — STEP/STP 转 URDF
- **类型：** site（Web 应用）
- **来源：** StackForce 轻量级机器人开发平台（stackforce.cc）
- **链接：** https://cad2urdf.stackforce.cc/upload
- **入库日期：** 2026-07-08
- **全文复核日期：** 2026-11-28
- **核对范围：** 抓取并通读 <https://cad2urdf.stackforce.cc/upload> 页面当前全部可访问 DOM 文案（41 条节点）；Web 服务前端与后端为闭源 SaaS 应用（源码未公开）。
- **通读：** 已完成
- **一句话说明：** 浏览器端多格式转 URDF 工具：支持本地解析 **STEP/STP（上限提升至 500MB）** 以及 **URDF/MJCF/USD/ZIP（上限 200MB，可预填 Link 与关节）**，再经 Link/关节配置与测试导出；支持本地 job 缓存与 StackForce `.cad2urdf` 工程导入。
- **沉淀到 wiki：** [stackforce](../../wiki/entities/stackforce.md)

---

## 站点要点（2026-11-28 全文更新核对）

### 三步工作流

1. **导入模型或资源包** — 选择 **STEP（最大 500MB）** 或 **URDF / MJCF / USD / 资源 ZIP（最大 200MB）**，也可从示例启动；URDF/MJCF/USD 会自动预填 Link 与关节（旧版 300MB 单 STEP 限制已更新为 STEP 500MB / 机器人包 200MB）。
2. **配置 Link / 关节** — 浏览器本地解析几何与既有结构后进入统一配置工作台调整连杆、关节与局部坐标系。
3. **测试和导出** — 交互验证后导出 URDF 相关资源包。

### 局限与开源边界

- 前端页面可访问全文说明，但底层转换脚本、WASM/几何引擎及服务端代码均**未开源**，仅可按黑盒站点实测行为与官方说明引用。

### 导入方式

| 模式 | 说明 |
|------|------|
| **导入本地 CAD** | 选择 STEP/STP；浏览器内解析，**原始 CAD 不上传** |
| **本地项目缓存** | 重新打开当前浏览器内已转换的 job |
| **导入 StackForce cad2urdf 工程** | 打开此前保存的工程文件 |

### 隐私与局限（页面强调）

- **300MB** 单文件上限
- STEP/STP **只作为 CAD 几何导入**；Link / Joint **仍需人工配置**（与全自动几何推断工具不同）
- 提供 **示例模型** 快速进入配置阶段

### 与历史入口

- 社区教程亦提及早期在线入口 `urdf.stackforce.cc`（STEP→URDF）；当前产品域名为 **cad2urdf.stackforce.cc**，与工作台 [stackforce-workbench.md](stackforce-workbench.md) 第二步 **SimReady / Isaac 工程导出** 衔接。

---

## 对 wiki 的映射

- 升格页面：[wiki/entities/stackforce.md](../../wiki/entities/stackforce.md)
- 交叉引用：[wiki/entities/step2urdf.md](../../wiki/entities/step2urdf.md)、[wiki/entities/urdf-studio.md](../../wiki/entities/urdf-studio.md)、[wiki/concepts/urdf-robot-description.md](../../wiki/concepts/urdf-robot-description.md)

## 参考链接

- CAD2URDF：<https://cad2urdf.stackforce.cc/upload>
- 工作台向导：<https://workbench.stackforce.cc/>
- 主站：<https://stackforce.cc/>
