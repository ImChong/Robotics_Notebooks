# una-auxme.github.io / ROS2SmolVLA（项目页）

- **标题：** ROS2SmolVLA: Enabling Small Vision-Language-Action Models for Integration into Industrial-Grade Lightweight Robots
- **类型：** site / project-page
- **URL：** <https://una-auxme.github.io/en/projects/ros2smolvla/>
- **机构：** 奥格斯堡大学机电一体化教席（Chair of Mechatronics / AuxMe）
- **配套论文：** [ROS2SmolVLA（arXiv:2608.23320）](https://arxiv.org/abs/2608.23320) — 归档见 [`sources/papers/ros2smolvla_arxiv_2608_23320.md`](../papers/ros2smolvla_arxiv_2608_23320.md)
- **代码入口：** <https://github.com/una-auxme/ros2smolvla_docker> — 归档见 [`sources/repos/ros2smolvla_docker.md`](../repos/ros2smolvla_docker.md)
- **数据 / 权重：** <https://huggingface.co/una-auxme/ROS2SmolVLA_ur10e_no_joints_crop_pick_place>
- **入库日期：** 2026-08-26

## 一句话摘要

奥格斯堡大学 **AuxMe / Chair of Mechatronics** 官方项目页：展示 **SmolVLA + ROS 2 + UR10e** 的本地小型 VLA 部署，并集中列出代码、数据与方法图。

## 公开信息要点（截至入库日）

- **作者：** Nils Mandischer、Noah Böckmann、Ludwig Holl、Lars Mikelsons（University of Augsburg, Chair of Mechatronics）。
- **页首卖点：** *enables industrial-grade lightweight robots to use small VLA models integrated with ROS 2*。
- **资源区：** **💻 Code**、**💾 Data**；论文写明 GitHub webpage 为推荐入口。
- **方法图：** Figure 1 给出 ROS2SmolVLA 数据流：自研节点（绿）与外部依赖（灰）；ROS 2 topic 与 LeRobot 接口分层。
- **组件清单（页上明文）：**
  - `ros2smolvla_interface_lerobot` — LeRobot ↔ ROS 2 经纪
  - `ros2smolvla_interface_camera` — 多相机异步摄入
  - `ros2smolvla_docker` — 容器化推理
  - `ros2smolvla_ur10e_sim` — Gazebo 数字孪生
  - `ros2smolvla_ur10e_real` — 真机采集 / 驱动
- **结果摘要（页上数字）：** 九场景总体成功率 **77.72%**；ID pick **78.33%**、成功抓取后 place **92.47%**；OOD pick **76.56%**、成功抓取后 place **61.22%**。强调颜色/形状泛化弱、蓝盒 drop 偏置、失败恢复约 **75%**。
- **会议：** Industry of the Future and Smart Manufacturing 2026。
- **BibTeX：** `@inproceedings{ros2smolvla2026,...}`。

## 开源核查（步骤 2.5）

**已开源。** 项目页同时给出 Code 与 Data；GitHub 组织仓与 Hugging Face 权重/数据集均可公开访问。不要写成「仅论文承诺」。

## 为何值得保留

- **非 PDF 入口：** 把五件套仓库与 HF collection 收成单一导航页。
- **工业读法：** 明确「本地小任务 / 合规」而不是「通才 VLA 榜单」。

## 关联资料

- 论文归档：[`sources/papers/ros2smolvla_arxiv_2608_23320.md`](../papers/ros2smolvla_arxiv_2608_23320.md)
- 代码仓库：[`sources/repos/ros2smolvla_docker.md`](../repos/ros2smolvla_docker.md)
