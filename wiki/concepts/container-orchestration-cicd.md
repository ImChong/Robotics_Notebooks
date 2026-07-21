---
type: concept
tags: [systems-engineering, docker, kubernetes, cicd, deployment, containers]
status: complete
updated: 2026-07-21
related:
  - ./observability-logs-metrics-tracing.md
  - ./software-security-basics.md
  - ./model-versioning-ota.md
  - ./edge-cloud-robotics.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_deploy_obs_security_primary_refs.md
summary: "容器与部署（Docker、Kubernetes、CI/CD）：训练/仿真 farm 与服务发布基座；明确不承担硬实时力矩环。"
---

# 容器编排与 CI/CD（Docker / Kubernetes / 持续交付）

## 一句话定义

**容器编排与 CI/CD** 用镜像固化运行环境，用声明式编排扩缩仿真与训练，用流水线把代码/模型晋升到可回滚的发布物。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCI | Open Container Initiative | 容器镜像与运行时标准 |
| K8s | Kubernetes | 容器编排系统 |
| CI/CD | Continuous Integration / Delivery | 持续集成与交付 |
| GPU | Graphics Processing Unit | 训练/仿真加速器 |
| SBOM | Software Bill of Materials | 软件物料清单 |

## 为什么重要

- 机器人学习栈依赖可复现环境（CUDA、驱动、Python）；镜像 digest 锁定是「能复现」的第一道门。
- 批训练、大规模 Sim、评测 farm 天然适合 K8s Job/Deployment。
- CI 可把门禁扩到 `make ci-preflight`、Sim2Sim 冒烟与模型签名。

## 核心原理

1. **Docker/OCI**：分层镜像、cgroup 限额、只读根文件系统。
2. **Kubernetes**：Pod 为调度单元；Service 暴露稳定端点；Job 跑有限任务。
3. **CI/CD**：构建 → 测试 → 签名 → 部署；制品含容器与模型版本。

## 工程实践

- 训练镜像与运行时镜像分离；运行时尽量瘦、无编译器。
- GPU 节点用 device plugin；资源请求写清，避免静默共享导致训练抖动。
- 流水线对 **策略权重** 走 registry 晋升，对 **机载固件** 走独立 OTA 通道——见 [模型版本与 OTA](./model-versioning-ota.md)。
- **不要** 假设普通 Linux 容器能提供 `SCHED_FIFO` 硬实时。

## 局限与风险

- 特权容器与挂载宿主机 Docker socket 是供应链风险。
- 节点内核与 NVIDIA 驱动漂移会导致「同镜像不同结果」。

## 关联页面

- [可观测性](./observability-logs-metrics-tracing.md)
- [软件安全基础](./software-security-basics.md)
- [边缘–云端协同](./edge-cloud-robotics.md)

## 参考来源

- [部署可观测安全一手资料](../../sources/sites/systems_engineering_deploy_obs_security_primary_refs.md)

## 推荐继续阅读

- Kubernetes Docs — Concepts
