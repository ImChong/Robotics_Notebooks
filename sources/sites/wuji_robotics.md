# 上海舞肌科技有限公司（舞肌科技 / Shanghai Wuji Technology）

- **类型**：公司 / 产业侧原始资料汇编（公开招聘文案、媒体报道、工商与备案线索）
- **收录日期**：2026-05-12
- **说明**：以下条目为 **2026-05-12** 可访问的公开页面摘录与链接；**规格与路线图以官方 datasheet / 合同为准**，招聘与新闻中的技术表述可能存在营销化措辞。

## 一句话

面向 **具身 AI 机器人** 的 **关节级动力部件** 供应商，公开叙事强调 **F 系列内转子永磁无刷电机** 的 **扭矩密度**，研发中心在上海、生产在江苏常州。

## 为什么值得保留

- 国内 **人形 / 灵巧手** 供应链中较常被讨论的 **小型高扭矩电机** 路线样本之一。
- 公开招聘与媒体稿中反复出现 **「200 g / 8 N·m 峰值扭矩」** 等可检索锚点，便于与仿真里的 **armature / 反射惯量**、关节包络选型对照（仍需独立实测验证）。

## 公开信息要点（按来源）

### 国家大学生就业服务平台 · 企业详情（公司简介全文）

- **链接**：<https://yazjy.ncss.cn/student/jobs/PkW5wLBq1LsLjuuFFFZhch/corp.html>
- **主体英文名**：Shanghai Wuji Technology Co Ltd
- **成立**：2019 年（页面文案；与工商公开信息一致量级）
- **定位**：机器人动力部件；研发、生产及应用生态；服务对象为 **具身 AI 机器人厂商**
- **地理**：研发中心 **上海虹桥商务区**；生产基地 **江苏省常州市**；注册地址页面显示为 **上海市嘉定区江桥镇金园三路 223 号 4 栋 1 楼 107–108**
- **融资**：「先后完成了数轮共计 **4000 万元** 的融资」
- **产品时间线**：「并于 **2022 年** 推出了首款 **F 系列电机** 产品」
- **技术表述（原文摘要）**：对传统 **内转子永磁无刷电机** 做结构重构与优化；宣称在 **电机自身约 200 g** 质量下可达 **8 N·m 峰值扭矩**，并强调高扭矩、低成本、易维护、兼容性与稳定性。

### 网易号 / 证券之星 · 投融资稿（含「Pan Motor」品牌叙事）

- **链接**：<https://www.163.com/dy/article/KELVSCVC051984TV.html>
- **日期**：2025-11-18（页面标注）
- **要点**：称天眼查公开信息整理 **B 轮融资**，投资方含 **华方股权基金、五源资本** 等；文中将电机产品线称为 **「Pan Motor」**，并重复 **200 g / 8 N·m 峰值扭矩** 与 **F 系列** 叙事。
- **注意**：文末声明内容由 **AI 辅助生成**，投资细节需以工商登记与官方披露为准。

### 官网 / ICP 线索

- 公开检索中常见备案域名 **`wujihand.com`**（与「灵巧手」拼音语义一致）；**2026-05-12** 抓取时站点曾返回 **503**，本资料不依赖其正文，仅作后续 curator 核验入口：  
  - <https://www.wujihand.com/>（若恢复可补抓产品页与白皮书链接）

### Wuji Hand 五指灵巧手（官方文档中心，2026-05-12 可访问）

- **文档阅读指引（总入口）**：<https://docs.wuji.tech/docs/zh/wuji-hand/latest/>  
  - 汇总 **GitHub**（含 [`mujoco-sim`](https://github.com/wuji-technology/mujoco-sim) 等）、**产品介绍**、**SDK（Python/C++）**、**ROS2**、**HMI**、**Retargeting**（文档中出现 Apple Vision Pro 配置）、**URDF/MJCF 描述与 MuJoCo 可视化** 等模块入口。
- **产品介绍（中文版）**：<https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview/>  
  - 明确 **Wuji Hand** 为舞肌自研 **20 主动自由度** 仿生灵巧手；官方参数表摘要（以页面为准）：自重 **580 ± 10 g（不含线缆）**，尺寸约 **201 mm × 75 mm × 50 mm**，指尖力 **15 N**，整手抓握最大静载 **10 kg**，控制频率 **1000 Hz × 20 轴**，通信 **USB 2.0**，工作电压 **12–20 V DC** 等。
- **产品介绍（英文版）**：<https://docs.wuji.tech/docs/en/wuji-hand/latest/overview/>（与中文版同构，便于对照术语）
- **开源 SDK 仓库（示例）**：<https://github.com/wuji-technology/wujihandpy> — Python 侧 `wujihandpy`（文档与 README 以仓库为准）

### 产业媒体盘点（二手归纳，参数以官方为准）

- **NE 时代 · 灵巧手产业链盘点**：<https://www.ne-time.cn/web/article/37087>  
  - 文中称舞肌推出 **连杆驱动** 灵巧手 **WUJI Hand**，并提及 **600 g、20 全主动自由度、直驱内置手指、指尖力 15 N、整手负载 10 kg、1 kHz 反馈、约 5 万元级定价** 等；另提到 **压阻触觉手套** 及宣传演示场景。**注意**：与官方文档中的 **自重 580 ± 10 g** 等表述存在细微差别，**工程引用应优先采信 docs.wuji.tech 参数表**。

### 招聘平台（岗位方向，非产品规格）

- 猎聘等企业页在检索结果中常列出 **深圳 / 北京** 等办公地与算法、触觉等岗位关键词；**具体 JD 以平台实时页面为准**，本库不逐条存档 JD 全文。

## 对 wiki 的映射

- 升格页面：[wiki/entities/wuji-robotics.md](../../wiki/entities/wuji-robotics.md)

## 参考链接（索引）

- 国家大学生就业服务平台企业详情：<https://yazjy.ncss.cn/student/jobs/PkW5wLBq1LsLjuuFFFZhch/corp.html>
- 网易号投融资稿：<https://www.163.com/dy/article/KELVSCVC051984TV.html>
- 水滴信用等企业信息页（工商照面）：<https://m.shuidi.cn/company-b307fa62c8a3e1dd67766cb44ef41401.html>
- Wuji Hand 文档中心（中文）：<https://docs.wuji.tech/docs/zh/wuji-hand/latest/>
- Wuji Hand 产品介绍（中文）：<https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview/>
- NE 时代灵巧手盘点（含舞肌段落）：<https://www.ne-time.cn/web/article/37087>
