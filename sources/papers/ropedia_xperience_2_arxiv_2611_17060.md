# Ropedia Xperience 2：联合扩展物理 AI 环境与行为（arXiv:2611.17060）

> 来源归档（paper）

- **英文标题：** Ropedia Xperience 2: A Gold Standard for Scaling Physical AI Environments and Behaviors
- **中文标题：** Ropedia Xperience 2：面向物理 AI 环境与行为规模化的高标准数据基座
- **类型：** paper / physical-ai / egocentric-data / 3d-reconstruction / real-to-sim
- **arXiv：** <https://arxiv.org/abs/2611.17060>
- **全文 HTML：** <https://arxiv.org/html/2611.17060v1>
- **项目页：** <https://ropedia.com/xperience/2-research/>
- **官方代码：** <https://github.com/Ropedia/Ropedia-Xperience-2>
- **数据集：** <https://huggingface.co/datasets/ropedia-ai/Ropedia-Xperience-2>
- **作者：** Hao Li 等 21 人；Ropedia Research Team
- **论文版本：** arXiv v1，2026-11-12
- **核对日期：** 2026-11-13
- **核对范围：** 通读 arXiv v1 HTML 全文（§1–§7、表 1–6、图注、附录 A–E，重点逐项核对附录 B 的配准与质量标准、附录 C 的场景组成、附录 D 的评测协议）；同时阅读官方项目页、GitHub README、配准文档与 Hugging Face 数据卡。
- **通读：** 已完成
- **一句话说明：** 同一次场景采集关联米制 3D 几何与多段第一视角动作，通过场景配准和 Real-to-Sim 等路径复用环境和行为。

## 核心事实（按不同统计口径区分）

| 项目 | 论文报告 | 统计口径 |
|------|----------|----------|
| 场景扫描 | 1,589 | 经过发布决策保留的扫描，未必全部已上传到 HF |
| 独立采集地点 | 596 | 更广泛采集清单，并非 1,589 个独立地点 |
| 任务定义 | 74 | 二级任务（Level 2） |
| 第一视角任务录像 | 4,230 条 / 131 小时 | 较广泛采集清单，非全部均已处理、配准及公开 |
| 场景类型 | 12 | 家居、工作/零售、户外场景 |
| 质量分级 | Perfect 148、Usable 934、Reuse 507 | 1,589 保留场景的分级；Reuse 包括部分扫描覆盖不足 |

注意：以上是论文的采集/发布统计，不代表 HF 仓库已完整下载开放。数据卡在核对日说明正在分批上传，第一批列出 126 个场景；详情见 [数据集归档](../datasets/ropedia-xperience-2.md)。

## 采集与加工流程

1. 对现场按「场景 Level 1 → 任务 Level 2」预先安排采集；环境保持相对静态。
2. 使用手持 KOSMO LiDAR 扫描仪（两鱼眼 + 两广角相机）获取米制场景；导出 COLMAP 位姿、三角网格和 3DGS。
3. 佩戴 HOMIE Gen 2（四路全局快门鱼眼、IMU、四声道音频），按场景逐项录制任务。
4. 四相机在统一 30 fps 帧索引上同步；双目估计深度，SLAM 恢复设备轨迹，MANO 估计手部，EgoAllo 与 SMPL-H 估计全身姿态；多模态模型生成任务、子任务、动作的分层语言标注。
5. 将 HOMIE 立体左目帧同扫描图像做 SIFT 匹配和 RANSAC/PnP 定位，网格反投影获得三维点，再以稳健 Umeyama 拟合记录坐标系到扫描坐标系的变换；使用留出帧、重力、深度 ICP 判断 PASS/WEAK/FAIL。

**观测与估计边界：** 头戴摄像头、音频、IMU 是传感器观测；手/身体姿态、SLAM、深度、网格均有估计误差；未观察到的腿部状态是推断值；语言标签由模型生成，不能等同人工真值。

## 实验与演示

- **三类 Benchmark：** 同一固定 20 场景（15 室内、5 室外），分别测深度与几何（11 个模型）、三维目标检测（7 种方法）、空间智能（12 个模型，500 个问答）。
- **深度/几何：** 输入密度分为 5 帧、11–100 帧、335–500 帧；没有单一模型统领全部指标。论文报告 π³ 在轨迹 ATE 和表面 F-score 上领先；部分优化式方法在 Dense 下 OOM。
- **3D 目标检测：** Boxer 的 AP25=24.69%、AP50=6.09%，其他方法 AP25 均低于 7%；评测只在 20 个有人工 3D 框标注的场景上进行。
- **空间问答：** Gemini-2.5-Pro 总体准确率 74.2%，HoliWorld-VL-4B 为 67.8%；距离等定量空间题仍困难。上述均为论文特定提示、场景和协议下的报告值。
- **Real-to-Sim：** 论文展示一个包含 217 个独立物体的物理可交互场景，渲染与扫描 3DGS 逐视角对照；此对照不构成独立几何真值。
- **Human-to-Robot Retargeting：** 在同一场景的四段录像共 10.3 分钟，轮式人形机器人轨迹共移动 96 米；手指末端跟踪误差中位数 0.40–0.55 cm，90% 关节速度/力矩阈值及 ZMP 等约束用于检查。
- **Virtual Teleoperation：** 通过头显/手输入在仿真厨房里遥操作人形机器人移动水壶，可在虚拟场景中重置再采集。
- **Long-Horizon Generation：** 单个八子任务任务，对 30 个留出初始状态全部完成；轨迹 3,869–4,255 控制步（30 Hz），单 worker 墙钟生成约 10–16 分钟。不是跨场景或真实机器人泛化成功率。

## 结论、误区与局限

- **主要主张：** 「环境几何 + 人类行为 + 可核验场景配准」是可重复使用的原始资产；Real-to-Sim、重定向、虚拟遥操作和程序化轨迹扩充是后续四条应用路径，并非论文证明了全自动物理 AI scaling law。
- 场景以家庭室内为主（1,389 / 1,589，约 87.4%），类目分布并不均匀；Reuse 场景占 31.9%，其局部范围未必完整。
- 数据发布分批进行，论文的「released scans」和 Hugging Face 实时可用文件数需分别报告。
- 头戴记录的单次全局刚体/相似变换无法修复长记录中的局部 SLAM 漂移。
- 论文展示的下游仿真、重定向及长程生成，不能冒充已发布、可独立复现的完整工具链。

## 对 wiki 的映射

- [Ropedia Xperience 2 主实体](../../wiki/entities/paper-ropedia-xperience-2.md)
- [Ropedia W0](../../wiki/entities/paper-ropedia-w0.md)：同团队跨本体世界模型的潜在数据需求对照
- [EgoXperience](../../wiki/entities/paper-egoxperience.md)：第一视角场景与手/物重建的相邻路线
