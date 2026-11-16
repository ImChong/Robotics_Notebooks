# TC-WM 源码核读

- **URL：** <https://github.com/MinghaoFu/TC-WM>
- **版本：** [5207220b94d4d9e7a31e80c0eeed6a90f8060885](https://github.com/MinghaoFu/TC-WM/tree/5207220b94d4d9e7a31e80c0eeed6a90f8060885)
- **核对范围：** 根README、planning/cem.py与mpc.py全文；plan.py的planning_main、PlanWorkspace初始化/规划入口，VWorldModel.rollout完整函数；递归树确认公开规划文件。未运行GPU。
- **通读：** 已完成

plan.py加载hydra.yaml、数据与checkpoint，实例化环境和PlanWorkspace，再由Hydra构造planner。CEM采样动作、WM.rollout、objective打分、topk更新均值/标准差；可选MPC执行前缀后从环境反馈重规划。CEM过程可调用evaluator提前停止，实验应记录这一设置。debug_dset_init会使用GT动作初始化，不能作为普通无提示规划。

MIT代码公开，README的checkpoint仍写will be released；树中无LDP命名实现或配置，尽管README列出conf/planner/ldp.yaml，故只将CEM/MPC路径画为已核实源码，不宣称论文LDP已完整交付。TC-Cosmos3未见单独实现/权重。
