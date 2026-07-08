# StackForce CAD2URDF（cad2urdf.stackforce.cc）

> 来源归档

- **标题：** StackForce CAD Importer — STEP/STP 转 URDF
- **类型：** site（Web 应用）
- **来源：** StackForce 轻量级机器人开发平台（stackforce.cc）
- **链接：** https://cad2urdf.stackforce.cc/upload
- **入库日期：** 2026-07-08
- **一句话说明：** 浏览器端 **STEP/STP→URDF** 转换器：CAD 几何在本地解析（不上传原始文件），再经 **Link/关节人工配置** 与测试导出；支持示例加载、本地 job 缓存与 StackForce 工程文件导入。
- **沉淀到 wiki：** [stackforce](../../wiki/entities/stackforce.md)

---

## 站点要点（2026-07-08 抓取）

### 三步工作流

1. **导入 CAD** — 选择本地 STEP/STP（最大 **300MB**）；或从示例开始
2. **配置 Link / 关节** — STEP 仅提供几何，连杆与关节需人工配置
3. **测试和导出** — 交互验证后导出 URDF 相关资源

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
