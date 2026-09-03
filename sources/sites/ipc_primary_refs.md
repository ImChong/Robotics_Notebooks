# 进程间通信（IPC）一手资料索引

> 来源归档（ingest）

- **标题：** POSIX/Linux IPC 机制与机器人中间件分层的一手依据
- **类型：** standard / man-page / book / official docs（合集）
- **入库日期：** 2026-09-03
- **一句话说明：** 为机器人主控 Linux 上「感知 / 运控 / 日志」分进程协作提供 IPC 机制谱系、POSIX 语义与实时选型边界的原始依据。
- **沉淀到 wiki：** 是 → [ipc-inter-process-communication](../../wiki/concepts/ipc-inter-process-communication.md)、[operating-system-basics](../../wiki/concepts/operating-system-basics.md)、[real-time-control-middleware-guide](../../wiki/queries/real-time-control-middleware-guide.md)

## 为什么值得保留

- 真机抖动常被归因于「策略不好」，但 **pipe 阻塞、共享内存缺同步、DDS 动态分配** 等 IPC 细节同样会拉长控制环尾部延迟。
- 机器人栈里 **LCM / DDS / 共享内存 / Unix 域套接字** 并存；需要一张从 **内核原语 → 中间件** 的一手对照表，避免把 HTTP/gRPC 误塞进 1 kHz 环。
- 本库此前仅在 [operating-system-basics](../../wiki/concepts/operating-system-basics.md) 与 [real-time-control-middleware-guide](../../wiki/queries/real-time-control-middleware-guide.md) 零散提及 IPC，缺少独立归档与概念页。

## 核心摘录

### 1) POSIX.1 / Linux man-pages — 经典 IPC 原语

| 机制 | 一手入口 | 要点 |
|------|----------|------|
| **管道 / FIFO** | [pipe(7)](https://man7.org/linux/man-pages/man7/pipe.7.html)、[pipe(2)](https://man7.org/linux/man-pages/man2/pipe.2.html)、[fifo(7)](https://man7.org/linux/man-pages/man7/fifo.7.html) | **单向字节流**、无消息边界；`PIPE_BUF`（Linux 4096 B）内写原子；满/空时 `read`/`write` 可阻塞；适合 **父子进程 / shell 管道**，不适合高频结构化状态 |
| **POSIX 共享内存** | [shm_overview(7)](https://man7.org/linux/man-pages/man7/shm_overview.7.html)、[shm_open(3)](https://man7.org/linux/man-pages/man3/shm_open.3.html)、[mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html) | `shm_open` + `mmap` 映射同一块物理页；**零拷贝**但须 **信号量/无锁队列** 同步；机器人本机 **传感环 ↔ 运控环** 常用 |
| **POSIX 消息队列** | [mq_overview(7)](https://man7.org/linux/man-pages/man7/mq_overview.7.html)、[mq_open(3)](https://man7.org/linux/man-pages/man3/mq_open.3.html) | 具名队列 `/somename`、**带优先级**、内核持久；比 SysV 消息队列接口更干净；适合 **命令/事件** 而非连续关节流 |
| **POSIX 信号量** | [sem_overview(7)](https://man7.org/linux/man-pages/man7/sem_overview.7.html) | 常作共享内存 / 环形缓冲区的 **互斥与背压**；实时路径优先 **无锁 SPSC 环** |
| **Unix 域套接字** | [unix(7)](https://man7.org/linux/man-pages/man7/unix.7.html)、[socketpair(2)](https://man7.org/linux/man-pages/man2/socketpair.2.html) | 本机 **流式/数据报**；可传 **SCM_RIGHTS 文件描述符**；ROS 2 / gRPC 本地传输常落在此层之上 |
| **Internet 套接字** | [socket(7)](https://man7.org/linux/man-pages/man7/socket.7.html) | TCP/UDP；跨主机；LCM 默认 **UDP 组播**、DDS **UDP/TCP** |
| **事件多路复用** | [epoll(7)](https://man7.org/linux/man-pages/man7/epoll.7.html)、[poll(2)](https://man7.org/linux/man-pages/man2/poll.2.html) | 服务面 **非阻塞 I/O**；**勿在硬实时控制线程** 内依赖 epoll 等可能阻塞/唤醒的路径 |

- **POSIX vs System V：** man-pages 明确 POSIX 共享内存与消息队列接口 **设计更简洁**；SysV（`shmget`/`msgget`）在旧代码库仍常见，新工程优先 POSIX。
- **对 wiki 的映射：** [ipc-inter-process-communication](../../wiki/concepts/ipc-inter-process-communication.md)

### 2) OSTEP — 进程、并发与 I/O 边界

- **来源：** Remzi H. Arpaci-Dusseau & Andrea C. Arpaci-Dusseau, *[Operating Systems: Three Easy Pieces](https://ostep.org/)*（免费教材）。
- **要点：**
  - **进程是隔离边界**：地址空间、文件描述符表独立；IPC 的存在正是因为进程不能随意读写彼此内存。
  - **管道语义**：字节流、阻塞读写、与 `fork`/`exec` 组合构成 shell 与简单生产者–消费者。
  - **并发与锁**：共享内存 IPC 必须配合同步原语；否则出现数据竞争与非确定性延迟（机器人运控的大敌）。
  - **持久化与 I/O**：文件系统、`fsync` 与管道/套接字同属「跨边界传数据」，日志进程与控制进程应分核分队列。
- **对 wiki 的映射：** [operating-system-basics](../../wiki/concepts/operating-system-basics.md)、[ipc-inter-process-communication](../../wiki/concepts/ipc-inter-process-communication.md)

### 3) Stevens — APUE 第 15–17 章（IPC 经典论述）

- **来源：** W. Richard Stevens, *Advanced Programming in the UNIX Environment*, 3rd ed.（APUE3）— 第 15 章（IPC 引言与管道）、第 16 章（System V IPC）、第 17 章（POSIX IPC：消息队列、信号量、共享内存）。
- **要点：**
  - 按 **带宽、延迟、持久性、命名空间、同步需求** 比较机制，而非只看 API 名字。
  - **文件描述符传递**（`sendmsg` + `SCM_RIGHTS`）是 Unix 域套接字相对管道的关键扩展。
  - 共享内存 **最快** 但 **最难正确**；消息队列与管道 **更安全** 但有拷贝开销。
- **对 wiki 的映射：** [ipc-inter-process-communication](../../wiki/concepts/ipc-inter-process-communication.md)

### 4) Beej's Guide to Unix IPC — 可操作的机制对照

- **来源：** Brian "Beej" Hall, *[Beej's Guide to Unix IPC](https://beej.us/guide/bgipc/)*（免费在线）。
- **要点：**
  - 以 **示例代码** 串联 pipe、FIFO、消息队列、信号量、共享内存、Unix/Internet 套接字。
  - 强调 **消息边界**（管道无、消息队列有）与 **全双工**（`socketpair`）差异。
  - 适合工程侧快速对照「该用哪种原语」。
- **对 wiki 的映射：** [ipc-inter-process-communication](../../wiki/concepts/ipc-inter-process-communication.md)

### 5) 机器人中间件层 — 一手官方定义（本库已有专页）

| 层 | 机制 | 一手入口 | 站内归档 |
|----|------|----------|----------|
| 本机零拷贝 | 共享内存 + 无锁环 | Linux `shm_overview(7)`；各项目自研（如 mimic-ipc、robot-io） | [real-time-control-middleware-guide](../../wiki/queries/real-time-control-middleware-guide.md) |
| 低延迟 pub/sub | LCM（UDP 组播） | [lcm-proj.github.io](https://lcm-proj.github.io/lcm/) | [lcm-proj-github-io.md](lcm-proj-github-io.md) |
| 系统集成 pub/sub | ROS 2 → DDS | [OMG DDS](omg-dds-spec.md) | [dds-communication](../../wiki/concepts/dds-communication.md) |
| 服务面 RPC | gRPC / ROS Service | [grpc.io](grpc-io-docs.md) | [remote-procedure-call](../../wiki/concepts/remote-procedure-call.md) |

- **分层原则（一手 + 本库实践）：** **1 kHz 关节流** → 共享内存 / LCM；**中高层状态** → ROS 2 Topic；**配置/标定** → RPC/Service；**跨机** → UDP 组播或 DDS，避免 TCP 重传进控制环。

## 推荐继续阅读（外部）

- Linux `ipc_namespaces(7)` — 容器内 IPC 隔离
- [LCM UDP Multicast Protocol](https://lcm-proj.github.io/lcm/content/lcm-udp-multicast-protocol-description.html)
- [ROS 2 About Different DDS Vendors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Different-Middleware-Vendors.html)

## 当前提炼状态

- [x] 摘要与 wiki 映射
- [x] 升格 [ipc-inter-process-communication](../../wiki/concepts/ipc-inter-process-communication.md)
