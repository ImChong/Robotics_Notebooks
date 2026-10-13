# geo-sleuth — 可核验的照片地理定位 Agent Skill

> 来源归档

- **标题：** geo-sleuth
- **类型：** GitHub 仓库 / Agent Skill
- **维护者：** Oldcircle
- **代码：** https://github.com/Oldcircle/geo-sleuth
- **主要入口：** README.md；skills/geo-sleuth/SKILL.md
- **安装：** npx skills add Oldcircle/geo-sleuth
- **许可：** MIT（以仓库 LICENSE 为准）
- **运行环境：** Python 3.10+、uv、curl；部分工具使用 Playwright、ML 依赖和模型权重
- **核查日期：** 2026-10-13
- **一句话说明：** 跨兼容 Agent 的照片地理定位 skill，使用脚本收集图像/地图/地形证据、枚举并排序候选，再由代理按实际运行产物进行交叉核验。
- **沉淀到 wiki：** [geo-sleuth 实体页](../../wiki/entities/geo-sleuth.md)

## 官方材料摘录

### 1. Agent skill 的分层职责

官方 SKILL.md 将工作分为候选决策、感知与计算、模型判断：board.py 记录候选和证据；多个专用脚本读图、查表、查询地物、匹配地形/影像；Agent 提出假设并判断排名靠前的候选。规则若可代码化则交给候选板执行，最终判断必须能指回实际会话命令与文件产物。

**对 wiki 的映射：** [实体页](../../wiki/entities/geo-sleuth.md)「核心设计」「流程总览」「源码运行时序图」。

### 2. 主要脚本与数据源

官方 data-sources 文档列出 EXIF/OCR、Baidu/Yandex 反向搜图、OSM/Overpass、太阳与阴影几何、DEM 地平线/山脊拟合、CLIP 卫星图排序、DINOv2 + SIFT 地面图像排序、候选管理与证据图等工具。区域包可加入国家相关的查询来源、提示、线索表和行政区数据。

**对 wiki 的映射：** [实体页](../../wiki/entities/geo-sleuth.md)「主要工具链」；把各类数据覆盖和坐标系视为结果边界。

### 3. 官方展示案例与精度读法

README 展示无 EXIF 的稻田/铁路桥/山脊照片，从 27,335 个初始候选逐步缩小，最终称机位误差半径约 ±2 m；也展示大柴旦沙漠与温哥华街道案例。上述是仓库文档所述示例，不是跨地区基准测试，也不能视为所有照片的精度承诺。

**对 wiki 的映射：** [实体页](../../wiki/entities/geo-sleuth.md)「复现与适用边界」「结论」。

### 4. 依赖、隐私与安全边界

README 建议先运行 doctor.py 检查本地工具和网络服务；反向搜图、卫星/街景与 DEM 查询依赖外部服务，服务可达不代表覆盖、可用性或处理权限。SKILL.md 对他人拍摄照片、私人住宅和未成年人图像规定先询问用途；定位不确定时要扩大范围、不要伪报精度。

**对 wiki 的映射：** [实体页](../../wiki/entities/geo-sleuth.md)「复现与适用边界」「结论」。

## 开源状态

官方仓库公开，标注 MIT，包含 skill 文件夹、Python 脚本、参考资料、案例和区域数据。读取仓库文档核验了安装和入口说明；本次未在本地克隆运行 doctor 或定位案例，因此不声称完成运行验证。

## 解释边界

- ±2 m 是 README 一个示例的自述结果，不代表普遍精度，也不是独立复现数据。
- 反向搜图和地理影像/地形数据可能涉及外部服务；敏感图像应先评估并可跳过反向搜图。
- OSM、街景、卫星图、高程和区域包的覆盖、时效及坐标系会影响候选结果。
- 它定位照片场景，不是机器人实时定位、SLAM 或经安全认证的导航系统。
