# Legged Lab DWAQ（Unitree G1）

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** Legged Lab DWAQ（Unitree G1）
- **类型：** repo
- **技术路线分组：** Locomotion与运动先验
- **链接：** https://gitee.com/chaomingsanhua/legged_lab
- **入库日期：** 2026-07-30
- **一句话说明：** G1 Actor读取100维当前本体观测，五帧历史经VAE估计速度和环境潜变量，再与PPO联合优化29维关节位置目标。代码属于DreamWaQ式盲走实现，缺少AdaBoot和实机通信链路。
- **开源状态（据主表）：** 已开源（以官方仓库 README 为准）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/methods/dreamwaq.md`](../../wiki/methods/dreamwaq.md)（不另建重复实体；合并入已有节点）

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [dreamwaq](../../wiki/methods/dreamwaq.md) — HMI 开源主表社区实现：gitee chaomingsanhua/legged_lab（Unitree G1 DWAQ 式盲走）
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
