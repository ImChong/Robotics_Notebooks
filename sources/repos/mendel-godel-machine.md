# Mendel Gödel Machine（MGM）官方代码与运行入口

- **仓库：** https://github.com/RealLcz/MGM
- **对应论文：** [arXiv:2608.07645](https://arxiv.org/abs/2608.07645)
- **项目主页：** https://reallcz.github.io/MGM/
- **开放状态：** 已开源；包含实际 Python 主入口、三种比较进化策略、评测和调度脚本。
- **核查日期：** 2026-11-20
- **核对范围：** 阅读 README、hgm.py 的 main()/sample()/expand()/TS_sample()，hgm_utils.py 的 sample_child()/eval_agent()、self_improve_step.py 三种 diagnose 函数、tree.py、config.yaml 和 scripts/slurm/mgm.slurm 主要调用链；未执行高成本模型实验。
- **通读：** 已完成
- **对 wiki 的映射：** [MGM 论文实体](../../wiki/entities/paper-mendel-godel-machine.md)

## 可验证源码链

| 文件 | 函数 / 入口 | 作用 |
|---|---|---|
| [hgm.py](https://github.com/RealLcz/MGM/blob/main/hgm.py) | main、TS_sample、sample、expand | 初始化进化树；使用 Thompson sampling 选择评测或扩展；检查 CM/RM/CH 资格 |
| [hgm_utils.py](https://github.com/RealLcz/MGM/blob/main/hgm_utils.py) | sample_child、eval_agent | 构建候选工作空间，选择诊断模式，运行编码 Agent 自修改，执行评测 |
| [self_improve_step.py](https://github.com/RealLcz/MGM/blob/main/self_improve_step.py) | diagnose_problem、diagnose_two_entries_same_commit、diagnose_shared_task_two_commits | 分别生成 A、B、C 的 problem_statement |
| [tree.py](https://github.com/RealLcz/MGM/blob/main/tree.py) | Node | 保存子代和任务评测记录 |
| [config.yaml](https://github.com/RealLcz/MGM/blob/main/config.yaml) | self_improve_strategy | 默认 A/B/C 权重 0.1/0.45/0.45；B 至少 2 次评测 |

## 官方复现命令

```bash
# 仓库根目录执行，需预配 Slurm、模型服务与容器
HGM_OUTPUT_DIR=<your_output_dir> sbatch scripts/slurm/mgm.slurm

# 预先启动兼容的模型服务并准备运行环境后，可直接执行
python hgm.py --config config.yaml --output_dir <your_output_dir> \
  --continue_from null --max_task_evals 200 \
  --self_improve_weight_a 0.1 --self_improve_weight_b 0.45 \
  --self_improve_weight_c 0.45
```

实际 hgm_utils.sample_child() 使用 docker_from_env() 和 build_hgm_container()，而 README 说明 Apptainer 运行模式无需 Docker daemon，存在部署指令与源码运行路径的不一致，应以将要运行的版本做独立核查；模型默认值也需核对 config.yaml 与 README 示例。生成代码必须在隔离环境执行。此处仅静态核查，**未复现论文分数**。

## 参考来源

- [仓库 README](https://github.com/RealLcz/MGM/blob/main/README.md)
- [论文附录的运行与参数说明](https://arxiv.org/html/2608.07645)
