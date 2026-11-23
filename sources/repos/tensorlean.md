# TensorLean/TensorLean：官方仓库与缺陷追踪

> 一手来源：项目维护者官方 GitHub README 与仓库根目录；当前**不是已开源的可运行验证框架**。

- **仓库：** https://github.com/TensorLean/TensorLean
- **项目主页：** https://tensorlean.ai/
- **研究方公告：** [CertiK 官方技术公告](../sites/tensorlean-certik-announcement.md)
- **知识节点：** [TensorLean](../../wiki/entities/tensorlean.md)
- **核对日期：** 2026-11-23
- **开放状态：** 仓库根目录仅见 README.md、assets/；README 声明论文、框架、PTX semantics、规格与证明“soon release”，未提供可运行源码或验证 CLI。
- **核对范围：** 完整阅读官方 README 的项目介绍、Release 声明、Correctness Bugs 全部缺陷条目及联系方式；核对仓库根目录实际文件列表。未阅读尚未公开的验证器源码、Lean 证明或项目主页不可获取的技术页面。
- **通读：** 已完成

## README 明确给出的规格与结果

开发者编写 Tensor-level Specification，TensorLean 将 GPU 算子已编译的 PTX 与该规格比对；对**满足规格前提的输入和所有执行**证明：无数据竞争、终止、功能正确（在浮点容差内）。

- 目标：生产级 NVIDIA GPU Kernel（PTX 层）。
- **149 个** open-source kernel 实例接受验证；其中 **102 个**来自 FlashAttention；**83 个**完成符合规格的证明。其余 **66 个**未获得同等证明，不能据此直接称每个实例对应一个独立缺陷。
- README 当前汇总 **39 个**已向上游开发者报告的 correctness bug；包括内存越界、数值问题、数据竞争、未初始化内存、未定义行为、死锁和编译错误。
- 开源与复现：README 仅为**将来发布的承诺**；不提供安装、运行入口、测试集、证明文件，不能独立复现验证流程。

## 具体可追溯的上游缺陷案例

1. [FlashAttention-4 #3195](https://github.com/Dao-AILab/flash-attention/issues/3195)、[PR #3196](https://github.com/Dao-AILab/flash-attention/pull/3196)：**compute_block_sparsity** 的 reduction_buffer 在 warp 之间重用时缺少必要的同步，可能把非空块判为空块。上游 Issue 说明对特定 mask_mod 路径生效，加入人为延迟可观察到错误；不能由此断言常规运行必现。
2. [DeepGEMM #514](https://github.com/deepseek-ai/DeepGEMM/issues/514)、[PR #515](https://github.com/deepseek-ai/DeepGEMM/pull/515)：SM90 grouped masked GEMM 当 M 不是 BLOCK_M 的整倍数且写至最后部分 block 时，跨 group 写错行并与其他 CTA 写入竞争，输出可能不确定。
3. [DeepGEMM #476](https://github.com/deepseek-ai/DeepGEMM/issues/476)、[PR #477](https://github.com/deepseek-ai/DeepGEMM/pull/477)：指定 FP8 einsum 路径遗漏加法参数 c，结果不符合调用期望。
4. [FlashAttention-3 #3193](https://github.com/Dao-AILab/flash-attention/issues/3193)：README 归为 Hopper forward producer warp barrier 同步缺陷所导致的潜在死锁。

以上为维护者所列具体 Issue / PR 的问题描述与适用条件，不等于这些 PR 均已合并或所有版本均受影响。

## 复现准备清单（尚不能执行）

- 等待官方发布框架源码、许可、PTX 语义、Lean Spec 与证明；
- 固定上游 kernel commit、编译器、目标架构与配置；
- 先定义规格前提与浮点误差范围，再做 PTX 级验证；
- 区分数学性质证明、部署实际执行路径与最终模型行为。

**源码运行时序图：** 不适用。官方仓库现无验证器的可执行入口；不能根据新闻稿虚构实际函数调用。
