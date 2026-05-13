# 文字生成 CAD / 对话式 CAD 工具（原始资料索引）

- **类型**：网站与在线产品（机械 CAD / AEC / API）汇编
- **收录日期**：2026-05-13
- **说明**：以下为 **2026-05-13** 可访问的公开页面链接与一句话定位；**定价、合规边界与导出能力以各产品当前文档为准**，本文件不做采购或合规建议。

## 一句话

**自然语言或对话**驱动 **可编辑 CAD 几何**（尤其 B-rep / STEP）或 **建筑平面**，与「纯三角网格生成」类工具在工程链路上不等价。

## 为什么值得保留

- 机器人硬件与夹具设计中，**STEP / 参数化模型** 仍是与加工、公差分析和仿真网格衔接的常见枢纽；LLM 维护者需要区分 **制造级 CAD** 与 **可视化网格**。
- **Zoo（KittyCAD）** 同时提供 **桌面设计器 + Zookeeper 对话代理 + 公开 Text-to-CAD API 与 Python SDK**，是可检索的工程样本。

## 主体产品：Zoo / KittyCAD

- **官网（产品总览）**：<https://zoo.dev/>  
  - 公开叙事：**Zoo Design Studio** 为 AI-native CAD；内置对话式代理 **Zookeeper**；强调 **B-rep**、**制造相关反馈**、可在指点式 / 代码 / 对话之间切换；提供 Mac / Windows / Linux 客户端入口。
- **Zookeeper（文字生成 CAD 试用入口，Web）**：<https://app.zoo.dev/?cmd=set-layout&groupId=application&layoutId=ttc>
- **Text-to-CAD 教程（Python + API 流程）**：<https://zoo.dev/docs/developer-tools/tutorials/text-to-cad>  
  - 文档说明：使用 **`kittycad`** Python 包、环境变量 **`ZOO_API_TOKEN`**；调用 `create_text_to_cad` 与轮询 `get_text_to_cad_part_for_user`；默认返回 **GLTF 与 STEP**（教程示例以 STEP 落盘）；失败时可能返回「提示必须清楚描述 CAD 模型」类错误，建议改写提示词。
- **API：从文本生成 CAD 模型**：<https://zoo.dev/docs/developer-tools/api/ml/generate-a-cad-model-from-text?lang=python>
- **ML API 总览**：<https://zoo.dev/docs/developer-tools/api/ml?lang=python>
- **KCL（Zoo 的参数化 CAD 语言）文档**：<https://docs.zoo.dev/docs/kcl>  
  - 与 Text-to-CAD 的关系：公开文档将 **KCL** 描述为 Zoo 模型背后的编程语言；Text-to-CAD 响应侧常见 **可审阅 / 可编辑的代码与模型** 组合（以 API 返回字段为准）。
- **信任与合规入口**：<https://trust.zoo.dev/>（站点首页宣称 **ITAR 分区**、**SOC 2 Type II** 等；工程使用须自行核验合同与数据驻留条款。）

## 同类或相邻赛道（便于对照，非穷尽）

> 下列条目按 **公开站点自我定位** 归类；**与 Zoo 是否构成竞争**取决于具体场景（机械零件 vs 建筑 vs 纯网格）。

### 建筑 / 户型（AEC 取向）

- **Maket.ai**：<https://www.maket.ai/> — AI 生成住宅平面、3D 可视化等；公开材料强调 **DWG / DXF** 类导出与建筑工作流（面向业主与营建侧，非通用机械 B-rep 零件库）。

### 公开宣称「文本 / API → STEP 或 B-rep」的第三方服务（须独立核验）

- **GrandpaCAD**：<https://grandpacad.com/> — 站点提供 **3D Model Generation API** 文档入口；公开叙事包含 **text-to-CAD** 与 **STEP** 等格式（以实时文档为准）。
- **PartWork AI**：<https://partwork.ai/> — 公开定位为 **AI 参数化 CAD**；站点材料强调 **B-rep / STEP** 与加工场景（以实时文档为准）。

### 「文字 / 图像 → 三角网格或场景资产」（通常不替代机械 CAD）

- **ModelsLab Text to 3D API**：<https://docs.modelslab.com/3d-api/text-to-3d> — 典型输出为 **OBJ / STL / PLY / GLB** 等网格格式文档入口；适合可视化与部分 3D 打印流程，**不**等同于有全量特征树的 STEP 零件。

## 对 wiki 的映射

- 升格页面：[wiki/concepts/text-to-cad.md](../../wiki/concepts/text-to-cad.md)

## 参考链接（索引）

- Zoo 官网：<https://zoo.dev/>
- Zoo Text-to-CAD 教程：<https://zoo.dev/docs/developer-tools/tutorials/text-to-cad>
- Zoo ML API（Python）：<https://zoo.dev/docs/developer-tools/api/ml?lang=python>
- KCL 文档：<https://docs.zoo.dev/docs/kcl>
- Maket.ai：<https://www.maket.ai/>
- GrandpaCAD：<https://grandpacad.com/>
- PartWork AI：<https://partwork.ai/>
- ModelsLab Text to 3D：<https://docs.modelslab.com/3d-api/text-to-3d>
