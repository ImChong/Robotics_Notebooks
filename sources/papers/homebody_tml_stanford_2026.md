# HomeBody: A Humanoid That Explores, Remembers, and Acts on Its Own（TML Stanford，2026）

> 来源归档（ingest）

- **标题：** HomeBody: A Humanoid That Explores, Remembers, and Acts on Its Own
- **类型：** project / humanoid / loco-manipulation / VLM orchestration / spatial memory / Real2Sim
- **项目页：** <https://tml.stanford.edu/homebody/>
- **GitHub：** <https://github.com/Stanford-TML/homebody>（README：**Code coming soon**）
- **作者：** Gio Huh（Caltech）；Cayden Gu、Takara E. Truong、C. Karen Liu、Guy Tevet（Stanford；Liu & Tevet equal advising）
- **机构：** 加州理工学院（Caltech）；斯坦福大学（Stanford University）The Movement Lab（TML）
- **发布：** 项目页标注 **September 2026**
- **入库日期：** 2026-09-28
- **一句话说明：** **Unitree G1** + 远程 **GPT Astra** 系统 2：探索采集 SLAM/视频/关节/路点 → **Astra Real2Sim agent** 建 **Isaac Sim** 数字孪生 → **可组合技能库**（Navigate / Pick / Place / Open drawer / Pick from drawer）结构化 tool call 执行；**Super Odometry + ICP** 定位；低层 **AMO** 协调行走与操作；真机长程厨房整理与模糊取药 demo，**无环境专属训练数据**。

## 开源状态（项目页 + GitHub 核查，2026-09-28）

- **待发布：** [Stanford-TML/homebody](https://github.com/Stanford-TML/homebody) README 仅 **「Code coming soon」** + overview 图；**无可运行训练/部署入口**。
- **项目页：** 含交互式 Real2Sim 3D 对比、技能视频与 FAQ；**未列** 权重 / 数据集下载。

## 摘要级要点

- **架构主张：** 质疑「System 2 VLM → 学习式 System 1 VLA → System 0 控制器」三段链；HomeBody 用 **前沿 VLM 直接编排可复用 motor skills**（plug-and-play skill library）。
- **持久空间记忆：** 探索阶段保留 keyframes / ego 观测于 **共享坐标系**（SLAM + 与仿真 ICP 对齐），支撑物体离开 ego 视场后的长程任务。
- **Real2Sim：** 相对纯人类录像，强调 **人形自采** SLAM 几何、关节、路点 + ego 视频，供 Astra agent 在 Isaac Sim 建 **几何+语义+视觉** 对齐孪生。
- **技能接口：** VLM 通过 **structured tool call** 选 skill + target（如 pick 的 0–1000 归一化图像点 + 手别；nav 的 2D 目标与朝向）；技能本地 **bounded retry** + 视觉伺服（SAM 2.1 + SAMURAI）；失败回传 VLM replan。
- **感知栈（FAQ）：** Fast-FoundationStereo 深度、分割、样条+IK+碰撞扫掠；抽屉开合并 **hook + 后退走**。
- **算力：** 技能/感知/规划在 **Razer Blade RTX 4090** 笔记本；Astra **远程** API。

## 核心论文摘录（MVP）

### 1) Explore → Real2Sim → Task

- **链接：** 项目页 Step 1–3
- **摘录要点：** 0.5× iPhone 视频、D435i、LiDAR+SLAM、关节、Astra 选路点；Real2Sim agent 输出 Isaac Sim 孪生；任务指令无 action-level 脚本。
- **对 wiki 的映射：**
  - [HomeBody](../../wiki/entities/paper-homebody.md) — 三阶段管线。
  - [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — 同属 VLM Real2Sim agent 谱系（对象/场景粒度不同）。

### 2) 技能库与 VLM 编排

- **链接：** 项目页 Implementation / FAQ
- **摘录要点：** Pick/Place/Navigate/Open drawer/Pick from drawer；可扩展接口；AMO 50 Hz 低层 + 250 Hz 臂/手。
- **对 wiki 的映射：**
  - [HomeBody](../../wiki/entities/paper-homebody.md) — 技能表与 control FAQ。
  - [GPT-Policy](../../wiki/entities/paper-gpt-policy.md) — 固定 VLM + tool 闭环对照。
  - [AMO](../../wiki/entities/paper-loco-manip-161-135-amo.md) — System 0 低层。

### 3) Demo 任务

- **链接：** 项目页 Long Horizon
- **摘录要点：** **Tidy kitchen**（多物体整理+丢弃）；**Retrieve medicine**（记忆 drawer、双手分工、顺带扔 carton）。
- **对 wiki 的映射：**
  - [HomeBody](../../wiki/entities/paper-homebody.md) — 结论与局限。
  - [Loco-Manipulation](../tasks/loco-manipulation.md) — 任务语境。

## BibTeX

```bibtex
@misc{huh2026homebody,
  author       = {Huh, Gio and Gu, Cayden and Truong, Takara E. and Liu, C. Karen and Tevet, Guy},
  title        = {{HomeBody}: A Humanoid That Explores, Remembers, and Acts on Its Own},
  year         = {2026},
  howpublished = {Project page},
  url          = {https://tml.stanford.edu/homebody/}
}
```

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-homebody.md](../../wiki/entities/paper-homebody.md)
