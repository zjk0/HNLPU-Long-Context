# HNLPU 项目长会话上下文总结

> 初次整理日期：2026-08-23
> 最近同步日期：2026-08-31
> 资料来源：指定 Codex JSONL 对话记录、对当前工作树的复核、2026-08-24 的两次 AttentionBuffer 增量提交、2026-08-27 至 28 的 Interconnect 初始化，以及 2026-08-29 至 30 的 CommunicationTask 与 direct Reduce 实现和测试
> 目的：为后续开发、汇报和新会话接续提供一份可独立阅读的项目上下文

## 1. 记录范围与结论摘要

本次读取的源文件为：

- 目录：/media/zjk/209287a7-18d2-4ab6-976a-fe134c6d1e54/kai/.codex/sessions/2026/07/01
- 文件：rollout-2026-07-01T11-35-49-019f1bbf-4dc0-7220-a429-3dcf7e63a821.jsonl
- 文件大小：13,915,449 bytes，约 13.9 MB
- 记录数：4,966 行 JSONL
- 用户回合：219
- 完成回合：219
- 中断事件：1
- 实际时间跨度：2026-07-01 至 2026-08-22
- 会话原始工作目录：/media/kai/Biggest Area/none/PhD/HNLPU-Long-Context
- 会话开始时 Git 状态：master 分支，提交 5cca6146539d62c77dcf858dc0aa7df6634341ef

虽然文件被保存在 2026/07/01 目录中，但它不是只有 7 月 1 日的一轮对话，而是一条持续到 8 月 22 日、围绕同一仓库不断演进的长会话。原记录中的绝对路径使用 /media/kai；后续整理使用过 /media/zjk 和 F:\none\PhD\HNLPU-Long-Context，当前工作树已移动到 /media/zjk/Biggest Area/none/PhD/Projects/HNLPU-Long-Context。本文因此主要使用仓库相对路径。

项目主线可以概括为：

1. 先建立“性能指标随上下文长度变化”的解析评估脚本。
2. 随后将目标升级为不执行真实 Transformer 数值计算、但能够模拟容量、资源竞争和周期依赖的 HNLPU 性能模拟器。
3. 逐步完成 Memory、Attention Buffer、HBM、KV Cache、Request、Config、Compute 和 Chip 的第一版。
4. 通过单层、多层和长上下文三份单芯片集成测试、AttentionBuffer 定向测试和 Interconnect 定向测试，验证计算流程、KV 隔离、Attention Buffer 向 HBM 溢出、碎片化 allocation、4×4 topology、CommunicationTask 数据语义和 direct Reduce 的通信时序与链路竞争。
5. Interconnect 已完成构造、拓扑、动态 link state、CommunicationTask 和 direct Reduce；broadcast、scatter、gather、all_reduce、all_gather、统一 execute、System、多请求与完整多芯片执行仍未完成。

## 2. 模拟器的最终定位

项目最终明确是“面向性能分析的资源与时序模拟器”，不是功能模拟器。

它不负责：

- 执行真实的 QK 乘法、Softmax、AV 乘法或 MoE 数值计算；
- 保存真实 K/V 张量；
- 生成真实 Token；
- 验证模型输出数值是否与 PyTorch 一致。

它负责：

- 记录某个请求、某层、某段 Token 的 KV Cache 在哪里；
- 记录 Attention Buffer、HBM、VEX、HNUnit 等资源忙到哪个周期；
- 计算操作的请求、开始、完成、等待和服务周期；
- 模拟容量不足、Bank 冲突、资源排队以及 HBM 回退；
- 最终统计延迟、吞吐、访存量、容量占用和资源利用率。

关键时间字段的语义是：

| 字段 | 含义 |
|---|---|
| request_cycle | 操作被上层发出的周期 |
| start_cycle | 资源和数据均可用后，操作真正开始的周期 |
| finish_cycle | 操作完成的周期 |
| ready_cycle | 数据完成写入并可被读取的周期 |
| busy_until_cycle | 某个硬件资源被占用到的周期 |
| wait_cycles | start_cycle 减 request_cycle |
| service_cycles | finish_cycle 减 start_cycle |
| total_latency_cycles | finish_cycle 减 request_cycle |

长期方向是 Trace-driven 的离散事件模拟。逐周期调用 tick 在超长上下文下会产生大量空转，因此对话中更倾向于由事件队列推进时间；但全局事件驱动器尚未实现。

## 3. 最终职责划分

对话中反复讨论后，模块边界最终收敛为：

~~~text
真实或合成 Trace
        ↓
Request
        ↓
Pipeline / Simulator                 尚未完成
        ↓
Chip
├── KVcacheManager
│   ├── AttentionBuffer
│   └── HBM
├── VEX
└── HNArray
    └── HNUnit

Interconnect                         拓扑、CommunicationTask 与 direct Reduce 已完成
System                               尚未完成
~~~

各对象职责如下：

- Request：表示一个请求及其生命周期状态。
- KVcacheBlock：描述“一块 KV Cache 是什么”。
- KVcacheManager：决定 Block 放在哪里，调用 Memory 完成存、读、释放，并维护索引。
- Memory：处理通用容量、allocation 和访问开销。
- AttentionBuffer：处理 Bank、1W1R、条带化布局和 Bank 级时序。
- HBM：使用统一容量、总带宽和共享忙碌状态进行粗粒度建模。
- ComputeTask：描述一项计算任务。
- VEX：执行 Attention 和向量/非线性操作的 timing model。
- HNUnit：表示一个特定 layer、weight type、expert 的固定权重计算资源。
- HNArray：保存多个 HNUnit，并把线性任务路由到正确单元。
- Chip：聚合本地存储、KV 管理和计算资源。
- Pipeline：未来负责依赖调度、流水推进、HBM 双缓冲和访存/计算重叠。
- CommunicationTask：描述某个 Request 的通信操作、参与 Chip、单份数据大小、root 和 group direction，本身不执行通信或保存 timing。
- Interconnect：只负责 Chip-to-Chip 通信；当前已建立 4×4 row/column topology、物理 link busy state 和 direct Reduce timing，后续继续扩展其他 collective。
- System：未来负责多 Chip 聚合、跨模块协调和全局调度。

最重要的边界是：Memory 不理解 request、layer、token 等 Transformer 语义；这些语义由 KVcacheBlock 和 KVcacheManager 管理。

## 4. 项目演进时间线

| 时间 | 阶段 | 主要结果 |
|---|---|---|
| 7月1日至2日 | 简化性能评估 | 完成四个上下文长度评估函数，生成 CSV 和曲线图，以 2K 吞吐做校准 |
| 7月10日至13日 | 架构设计 | 确认状态/周期模拟方向，拆分 Chip、Request、KV Cache、Memory、Compute、Pipeline 等职责 |
| 7月18日至27日 | 存储模型 | 完成 Memory、AttentionBuffer、HBM 及 allocate/write/read/free 闭环和内置测试 |
| 7月28日至8月3日 | KV Cache 与 Request | 完成 KVcacheBlock、KVcacheManager、AB→HBM 回退、混合读取、Request 数据模型 |
| 8月4日至10日 | Config、Compute、Chip | 完成 YAML 配置接入、初版计算资源、Chip 聚合和首个 Prefill/Decode 联合测试 |
| 8月12日至15日 | 边界测试 | 增加 runtime overrides、长上下文 HBM 溢出测试和三层 KV 隔离测试 |
| 8月17日至22日 | 计算模型细化 | 引入 HNUnit，重构 HNArray，扩展 ComputeTask/VEX，并迁移三份集成测试 |
| 8月24日 | AttentionBuffer 分配修正 | 保留 balanced 优先路径，增加 single-group 与 multi-group fallback、统一 bank_groups record，并把 fallback 的 bank 内策略细化为 access-width circular round-robin |
| 8月27日至28日 | Interconnect 初始化 | 完成参数校验、单位换算、4×4 row/column groups、48条无向物理 link 的 busy state、配置接入和18项定向测试；未实现具体通信 timing |
| 8月29日至30日 | CommunicationTask 与 direct Reduce | 新增纯通信任务容器；实现 direct Reduce 的 topology/link 校验、并行 phase、链路竞争、原子状态更新和16项新增 timing case，完整测试增至85项 |

### 4.1 第一阶段：解析性能模型

最初完成了四个函数：

- throughput_eval
- kv_cache_size_eval
- memory_access_latency_eval
- attention_compute_cost_eval

当前工作树中的对应脚本是 [performance_eval_simplify_codex.py](simplified_eval/performance_eval_simplify_codex.py)。最初记录中的文件名是 performance_eval_simpify.py，存在 simpify 的拼写。

主要公式如下。

KV Cache 总容量：

~~~text
KV_bytes =
    2
    × context_length
    × num_kv_heads
    × head_dim
    × num_layers
    × bytes_per_element
    × batch_size
~~~

其中系数 2 表示 K 和 V。

Attention 简化工作量：

~~~text
attention_ops =
    4
    × context_length
    × num_q_heads
    × head_dim
    × num_layers
    × batch_size
~~~

吞吐模型先用论文中 2K 上下文、batch 216、249,960 tokens/s 的点反推一步执行时间，再拆成：

~~~text
固定或弱相关开销
+ 随上下文长度线性增长的 Attention 开销
+ 超过 256K 后逐步暴露的 HBM stall
~~~

内存 stall 在 256K 及以前设为 0，在 256K 到 512K 之间线性增长，到 512K 时使 stall 占总时间约 10.7%。如果 stall 比例为 r，非 stall 时间为 T，则：

~~~text
stall_time = T × r / (1 - r)
~~~

这套模型适合快速扫描趋势，不是论文的精确 cycle simulator。它对校准点和人为假设非常敏感。

需要注意一个配置演进差异：

- 7 月 1 日首次实现时，2K Attention 时间占比被设为 0.25，并明确标注为假设。
- 会话末端读取到的配置以及当前 [hnlpu_config.yaml](hnlpu_config.yaml) 中，该值为 0.0055。
- 记录中没有一条清晰的助手修改说明解释 0.25 到 0.0055 的来源，因此后续实验必须记录实际使用值，不能混用两者。

### 4.2 第二阶段：从公式转向状态和周期

这一阶段确定了几个基本原则：

- 不保存真实张量，只保存描述符和完成周期。
- 静态硬件配置、动态资源状态、逻辑数据状态和性能统计分开。
- 全局 current_token 无法支撑 batch 和流水，请求进度应属于 Request。
- allocation 不能只传 size，还必须有唯一 ID；KV 语义则由 Block 保存。
- 分配失败原则上应是原子的：不能留下部分容量、错误游标或半发布索引。
- Chip 与 Memory、KVcacheManager 是组合关系，不是继承关系。

## 5. 论文参数、派生参数与人为假设

以下表格区分了“论文或项目明确值”和“模拟器策略/假设”。这一区分对后续论文实验非常重要。

| 参数 | 当前取值 | 来源或性质 |
|---|---:|---|
| 模型 | gpt-oss-120b | 项目目标模型 |
| Transformer layers | 36 | 模型配置 |
| hidden size | 2880 | 模型配置 |
| Q heads | 64 | 模型配置 |
| KV heads | 8 | 模型配置 |
| head dimension | 64 | 模型配置 |
| experts | 128，Top-4 | 模型配置 |
| HNLPU 拓扑 | 4×4，共16颗 Chip | 论文/项目设定 |
| Attention 映射 | 每列处理16个 Q heads；KV 按上下文维度在4行间切分，每颗相关 Chip 约处理 L/4 | 论文数据流 |
| stages per layer | 6 | 论文数据流 |
| pipeline slots / max batch | 216 | 36层 × 6阶段，论文/校准设定 |
| 时钟 | 1 GHz | 论文实现值 |
| 2K 吞吐 | 249,960 tokens/s | 论文校准点 |
| Attention Buffer | 20,000 banks × 16,000 bytes = 320 MB/Chip | 论文值，按十进制单位 |
| Bank 端口 | 1W1R | 论文值 |
| Bank 访问宽度 | 32 bit，即4 bytes/cycle | 论文值 |
| AB 固定延迟 | 3 cycles | 论文值 |
| AB 峰值带宽 | 80 TB/s | 由 Bank 数、宽度和频率推导，也与论文报告一致 |
| HBM | 8 stacks × 24 GB = 192 GB/Chip | 论文值 |
| CXL | 128 GB/s/link | 论文/配置值 |
| CXL 固定延迟 | 100 ns | 当前配置值；对话未清楚确认其论文来源 |
| 256K 前 HBM stall | 近似隐藏 | 论文 Fig.14 趋势 |
| 512K HBM stall | 约10.7% | 论文 Fig.14 |
| Bank group | 32 banks/group，共625组 | 模拟器映射策略，不是论文明确结构 |
| KV dtype | 2 bytes | 假设 |
| HBM 带宽 | 6.4 TB/s/Chip | 假设 |
| HBM 固定延迟 | 100 ns | 假设 |
| HNArray 各 weight type | 10 cycles | 假设 |
| SwiGLU / RMSNorm / residual / sampling | 20 / 10 / 5 / 50 cycles | 假设 |
| VEX Attention 工作量 | q_length × kv_length，再除以32 | 第一版等价工作量，不是严格物理 cached-head 公式 |
| Expert 到 Chip 映射 | row-major 均分，每 Chip 8个 | 当前模拟假设 |
| VEX 资源状态 | 每 layer 一个 busy state | 用户指定的当前抽象，需核对物理依据 |

## 6. 各模块的最终设计与实现状态

### 6.1 [memory.py](hnlpu_sim/memory.py)

#### Memory 基类

第一版通用能力包括：

- 总容量、带宽和固定访问延迟；
- usage、剩余容量和使用率查询；
- allocation ID 到容量记录的映射；
- allocate、free 和聚合访问耗时；
- 参数与 ID 校验；
- 状态一致性检查。

错误语义逐步统一为：

- 类型错误：TypeError；
- 非法值、空 ID、重复 ID、不存在 ID：ValueError；
- 内部账目不一致：RuntimeError；
- 正常容量不足：返回 False，且不修改状态。

用户明确偏好不过度封装，因此后期减少了大量只读 property 和下划线私有字段，主要依靠约定、一致性检查和测试保证正确性。

#### AttentionBuffer

默认结构：

~~~text
20,000 banks
16,000 bytes/bank
32 banks/group
625 groups
512,000 bytes/group
1W1R
4 bytes/cycle
3-cycle latency
1 GHz
~~~

分配策略：

1. 从 next_bank_group_id 开始按 group 轮询搜索。
2. Stage 1 保留 balanced single-group allocation：按 base_num_per_bank 与 remainder 将数据均匀条带化；group 总容量和所有目标 Bank 均能容纳时优先采用。
3. Stage 2 在 Stage 1 全部失败后，寻找第一个总剩余容量足够的 group；从该 group 的 next offset 开始，以 access_width_byte 为单次上限在 Bank 间 circular round-robin，直到完整 allocation 被规划完毕。
4. Stage 3 在没有单个 group 能完整容纳时，从 next_bank_group_id 开始跨 group 分配；每个 group 内仍采用相同的 access-width circular round-robin，允许一个 allocation 跨多个 Bank group。
5. Bank 剩余容量不足 access_width_byte 但大于0时，按真实剩余字节分配，不浪费 1 至 3 byte tail；多轮访问同一 Bank 时在临时 plan 中累计其总字节数。
6. Stage 1、2、3 的搜索和规划均不修改真实状态；只有 Bank 与 group 两级计划的总和都等于 allocate_size_byte 后，才统一提交 usage、索引和 cursor。
7. 在内部状态一致的前提下，只要 Attention Buffer 总剩余容量足够，Stage 3 就应保证 allocation 成功；False 仅表示总容量不足，随后 KVcacheManager 才将整个 KV Block 回退到 HBM。

round-robin cursor 规则为：Stage 1 保留原行为；Stage 2 把已使用 group 的 next offset 移到最后一次实际分配 Bank 的下一个位置，并把 next group 移到当前 group 的下一个；Stage 3 对每个已使用 group 分别更新 next offset，最终 next group 指向最后一个实际使用 group 的下一个。fallback 的 circular traversal 如果完整一轮没有分配任何字节，会抛出 RuntimeError，避免死循环或静默接受不一致状态。

主要动态状态：

- usage_byte；
- allocate_info；
- bank_usage_byte；
- bank_group_usage_byte；
- next_bank_group_id；
- next_bank_group_offset；
- bank_read_busy_until_cycle；
- bank_write_busy_until_cycle。

allocation 记录结构为：

~~~text
allocation_id:
    size: 总逻辑字节数
    bank_groups:
        group_id: 该allocation在此group中的字节数
    bank:
        bank_id: 该allocation在此bank中的字节数
    ready_cycle: write完成后加入
~~~

Stage 1 的单 group allocation 也统一使用 bank_groups 字典，而不再使用单值 bank_group 字段。check_consistency 会分别重建 Bank 与 group 用量，并检查两级映射都与 allocation size 和真实 usage 一致；free_memory 则遍历 bank_groups 与 bank 映射释放容量。read/write 继续只消费 bank 映射，因此无需为跨 group allocation 重新设计。

读写时序：

- 同 Bank 的多个 read 竞争读端口；
- 同 Bank 的多个 write 竞争写端口；
- 读端口和写端口独立，体现 1W1R；
- 不同 Bank 可并行；
- write 完成后增加 ready_cycle；
- 未 write 的 allocation 不能 read；
- read 到达早于 ready_cycle 时会等待；
- 端口发射结束和流水尾部数据完成被区分。

非 4-byte 对齐的 allocation 也被支持：容量按真实字节记录，访问次数按 4-byte 宽度向上取整。多个 allocation 在同一 Bank 时，需要分别向上取整后累加 issue cycles，不能先合并字节再向上取整，否则会低估访问次数。

Bank conflict 当前通过端口 busy-until 引起的等待体现，但没有独立统计冲突次数、请求队列、真实地址冲突或动态重映射。

#### HBM

HBM 第一版是统一容量池和共享带宽资源：

- 8 × 24 GB，总计192 GB；
- read 可聚合多个 allocation；
- read/write 使用带宽和固定延迟换算周期；
- write 后设置 ready_cycle；
- 所有读写共享一个 busy_until_cycle；
- 不模拟 Stack、Channel、Pseudo-channel 或地址级并行。

传输周期近似为：

~~~text
transfer_cycles =
    ceil(access_bytes × clock_frequency / bandwidth_bytes_per_second)
~~~

单一 busy state 是偏保守的串行模型。论文中的 HBM 双缓冲没有直接塞进 HBM.read；对话最终决定以后建立独立 DoubleBuffer 或 HBMStagingBuffer，并由 Pipeline 调度“一个 Buffer 计算、另一个 Buffer 加载”的重叠。

### 6.2 [kv_cache.py](hnlpu_sim/kv_cache.py)

#### KVcacheBlock

Block 是轻量数据容器，当前字段包括：

- block_id；
- request_id；
- layer_id；
- first_token_position；
- num_tokens；
- size_byte；
- token_stride，当前固定为4；
- chip_id；
- storage_location；
- allocate_id。

没有单独 state 字段。未放置、已分配、已写入等状态由 storage_location、allocate_id 和 Memory 中的 ready_cycle 推断。

#### KVcacheManager

设计为每颗 Chip 一个 Manager，而不是每个 Request 或 Layer 一个。

三个索引是：

~~~text
kv_cache_blocks:
    block_id -> KVcacheBlock

request_blocks:
    request_id -> set(block_id)

request_layer_blocks:
    (request_id, layer_id) -> 按first_token_position排序的block_id列表
~~~

已实现的生命周期：

- store_kv_block：优先 AB，失败后尝试 HBM；完成 allocate + write + 索引发布；
- read_kv_blocks：支持 AB-only、HBM-only 和 AB+HBM 混合读取；
- free_kv_block：释放底层 Memory 并同步删除三个索引；
- free_request：释放一个请求的全部 Block；
- check_consistency / ensure_consistent：校验 Manager、Block 和 Memory 映射。

store 的设计接近事务：

1. 分配前检查参数、Chip 归属和重复状态。
2. 优先 Attention Buffer，失败后回退 HBM。
3. write 成功后才修改 Block 并发布 Manager 索引。
4. write 或索引发布异常时释放刚才的 allocation 并回滚。

统一读取返回：

~~~text
request_cycle
start_cycle
finish_cycle
wait_cycles
service_cycles
total_latency_cycles
total_read_size_byte
attention_buffer_result
hbm_result
~~~

Manager 不强制一次 read 的 Block 属于同一 Request 或 Layer；调用者传入明确 Block ID，并负责选择正确集合。

一致性检查可通过构造参数全局开启或关闭。开发和测试阶段适合开启；长上下文性能模拟中默认关闭，以避免每次操作遍历大量 Block 和 20,000 个 Bank。

当前限制：

- 只有“新 Block 在 AB 放不下时改存 HBM”；
- 没有主动迁移、淘汰、换入换出；
- 混合 AB/HBM read 不是严格原子操作；
- free_request 中途失败时，之前已释放的 Block 不回滚；
- 单个 KV Block 仍不会在 Attention Buffer 与 HBM 两个 memory tier 之间拆分；AB 总剩余容量不足时，整个 Block 才回退到 HBM。

### 6.3 [request.py](hnlpu_sim/request.py)

Request 目前是数据容器，保存：

- request_id；
- input_token_num；
- output_token_num；
- arrival_cycle；
- generated_token_num；
- current_token_position；
- status；
- phase；
- start_cycle；
- finish_cycle。

request_id 由外部生成，便于复现实验并沿用数据集 ID。

output_token_num 在真实在线推理开始前通常未知，但本项目选择 Trace-driven 回放，因此可以使用真实 Trace 中已经观察到的输出长度作为模拟终止条件。

当前 Request 没有完整参数校验，也没有 start、advance_token、finish 等状态更新方法；测试和上层代码目前直接修改属性。

### 6.4 [config.py](hnlpu_sim/config.py) 与 [hnlpu_config.yaml](hnlpu_config.yaml)

Config 已实现：

- 读取 YAML；
- 校验 model、hnlpu、vex、memory、interconnect、eval 分组；
- 校验类型、取值范围和关键关系；
- 校验 4×4 拓扑、216 个流水槽、Expert 可均分、AB 容量与 Bank 乘积一致；
- 把 GHz、GB、TB/s、ns 转换为 Hz、byte、byte/s、秒；
- 支持运行时 overrides。

overrides 的格式是两层字典，并使用 section-level dict.update。它不是递归合并：

~~~text
覆盖 memory 中两个字段：
    其他 memory 字段保留

覆盖 eval.context_lengths：
    整个 context_lengths 子字典被替换
~~~

为构造 64,000-byte 的长上下文测试 Buffer，attention_buffer_mb_per_chip 从“必须是整数”放宽为“必须是正数”，必填和容量一致性检查仍保留。

### 6.5 [compute.py](hnlpu_sim/compute.py)

#### ComputeTask

主要字段：

- request_id；
- task_type；
- workload；
- layer_id；
- weight_type；
- expert_id。

最终主要任务类别：

- linear；
- attention；
- vector。

旧 projection 会自动规范化为 linear。

#### HNUnit 与 HNArray

HNArray 只处理带权重的线性任务：

- 非专家权重：q、k、v、xo、router；
- 专家权重：up、gate、down。

内部资源索引：

~~~text
(layer_id, weight_type, expert_id) -> HNUnit
~~~

每个 HNUnit 有独立 busy_until_cycle。因此不同 layer、不同 weight type 或不同 expert 可以并行；同一个 Unit 上的任务排队。

Chip 按 row-major 线性 ID 均分 128 个 Expert。16 个 Chip 时每颗分到8个，每颗创建：

~~~text
36 layers × (5个非专家Unit + 3类专家权重 × 8个Expert)
= 1,044 HNUnit
~~~

八类线性操作当前都使用10-cycle固定延迟，只是接口和资源竞争模型，不是论文精确测量值。

#### VEX

VEX 负责：

- Attention；
- vector 类型中的 SwiGLU、RMSNorm、residual、sampling；
- Softmax 在职责上属于 VEX，但当前没有单独 timing 参数，因此显式未实现。

Attention 第一版服务周期：

~~~text
equivalent_work = q_length × kv_length
service_cycles = max(1, ceil(equivalent_work / 32))
~~~

它被明确称为等价工作量，不能解释成论文严格定义的 cached KV-head 数量。

VEX 当前使用按 Layer 的 busy_until_cycle 列表：

- 同一 Layer 的 Attention 和 vector 操作串行；
- 不同 Layer 可以并行。

这项设计来自用户最后的明确要求，但早先分析曾提醒：如果真实硬件每颗 Chip 只有一个共享 VEX，按 Layer 独立会凭空增加资源。后续必须根据论文或 RTL 重新确认。

### 6.6 [chip.py](hnlpu_sim/chip.py)

Chip 从 Config 创建并持有：

- AttentionBuffer；
- HBM；
- KVcacheManager；
- VEX；
- HNArray。

Manager 引用的 AttentionBuffer 和 HBM 与 Chip 自身持有的是同一对象。Chip 还负责：

- 校验 row、column；
- 用二元组保存 chip_id；
- 接入模型层数和 VEX 配置；
- 按当前 Chip 坐标分配本地 Expert；
- 用各 weight type 的 latency 映射创建 HNArray。

### 6.7 [interconnect.py](hnlpu_sim/interconnect.py)

Interconnect 的范围被明确限定为 HNLPU Chip 之间的通信，不负责 Attention Buffer 与 HBM、Memory 与 Compute 或其他片内数据搬运。当前构造函数接收：

- chip_grid_rows、chip_grid_cols；
- link_bandwidth_GBps、link_latency_ns；
- clock_frequency_hz；
- collective_algorithms 字典。

行列数必须是大于0且非 bool 的 int；带宽和时钟必须是大于0且非 bool 的数值；link latency 必须是大于等于0且非 bool 的数值。collective_algorithms 必须包含 broadcast、reduce、scatter、gather、all_reduce 和 all_gather，且每个值都是非空字符串。算法名称没有被限制为 direct，以便后续加入 ring、tree、recursive_doubling 等抽象。

构造过程保存原始参数，并得到：

~~~text
num_chips = chip_grid_rows × chip_grid_cols
link_bandwidth_byte_per_s = link_bandwidth_GBps × 10^9
link_latency_cycles = ceil(link_latency_ns × clock_frequency_hz / 10^9)
~~~

collective_algorithms 使用浅拷贝，避免调用方之后修改原字典而改变 Interconnect 的内部配置。4×4、128 GB/s、100 ns 和 1 GHz 时，num_chips 为16，带宽为128,000,000,000 byte/s，固定延迟为100 cycles。

Topology 使用 row-major 线性 Chip ID：

~~~text
chip_id = row × chip_grid_cols + column

row_groups[0] = [0, 1, 2, 3]
column_groups[0] = [0, 4, 8, 12]
~~~

_build_links() 依次遍历所有 row group 和 column group，只组合每组中 source 之后的 destination，因此同一行或同一列内的每对 Chip 各建立一次直接物理 link。key 统一规范化为 `(min_chip_id, max_chip_id)`，不会同时创建 `(0, 1)` 和 `(1, 0)`。不同 row 且不同 column 的 Chip 之间没有直接 link。4×4 topology 最终共有48条无向 link：4行各6条加4列各6条。

每个 key 在 link_busy_until_cycle 中初始化为0。当前模拟抽象让两个通信方向共享同一条物理 link 的 busy state，尚不区分 full-duplex 的方向资源；如果以后需要独立方向状态，应在 timing model 设计阶段单独扩展。

CommunicationTask 是纯数据容器，保存 request_id、operation、participants、data_size_byte、source_chip、destination_chip 和 direction。它严格校验六种 operation 的 root 语义，并复制 participants；但不选择算法、不验证具体 topology、不查询 link、不保存或计算 timing。algorithm 继续由 Interconnect.collective_algorithms 决定。

Interconnect.reduce() 当前实现 direct Reduce：每个非 destination participant 通过一条直接物理 link 发送一份完整 partial result，canonical link key 仍为 `(min(src, dst), max(src, dst))`。所有 required links 先完整验证，再等待其中最晚的 busy_until_cycle，并在同一个 phase 中并行传输。单 link service 为 fixed link latency 加向上取整的 serialization cycles；成功后只更新实际使用的 links。单 participant 是0-cycle no-op。

Reduce arithmetic latency 当前假设为0或与通信完全重叠；不同 direct links 并行、统一等待后启动、link 在 fixed latency 加 serialization 期间保持 busy，均属于 simulator abstraction，不是论文精确结论。当前不做 multi-hop，也没有实现 tree 等其他 Reduce 算法。broadcast、scatter、gather、all_reduce、all_gather 和统一 execute 仍未实现。

## 7. 发现并修复过的关键问题

### 7.1 Memory 与 Attention Buffer

对话中曾发现并修复：

- f-string 语法错误；
- 负数 allocation 破坏 usage；
- free 同时接收 ID 和 size，可能互相矛盾；
- 分配失败却改变容量或 next-group 游标；
- 普通除法产生浮点数并传入 range；
- 最后一个 group 的切片边界错误；
- 成功路径忘记返回 True；
- 只检查 group 总容量，不检查目标 Bank；
- 某个候选 group 的 Bank 放不下后没有继续搜索；
- continue / break 层级错误导致错误提交；
- 一致性检查只在操作后执行；
- 非对齐 allocation 合并读取时低估 issue cycles；
- group 总容量足够但 Bank 剩余容量不均匀时，balanced striping 错误拒绝该 group；
- AB 总容量足够但没有单个 group 能完整容纳时，allocation 错误返回 False 并提前 spill 到 HBM；
- 旧单值 bank_group allocation record 无法表示跨 group allocation，现已统一为 bank_groups 字典；
- fallback 最初按 Bank 逐个填满，数据过度集中；现改为每次最多分配 access_width_byte 的 circular round-robin，并正确累计同一 Bank 的多轮计划；
- fallback circular traversal 未扣除尚未 commit 的同 Bank planned bytes 时可能重复使用容量，现已在计算 Bank remaining 时一并扣除；
- allocation 搜索阶段提前修改容量或 cursor 可能破坏失败原子性，现统一先生成并校验 plan，再一次性 commit；
- 高频一致性检查可能成为模拟器自身性能瓶颈。

### 7.2 KVcacheManager

曾修复：

- storage_location 字符串不统一；
- 同一 Block 被重复分到 AB 和 HBM；
- write 失败后没有释放刚分配的空间；
- 过早修改 Block 或索引，失败后留下半完成状态；
- request/layer 索引保存对象而不是 ID；
- 同层 Block 未按 Token 位置排序；
- 混合读取的 wait/service 字段直接取最大值，导致时间关系不一致；
- except pass 吞掉真实错误；
- 三种读取路径返回结构不统一；
- 热路径对列表反复执行 in，潜在 O(n²)；
- Memory 已释放但 Manager 索引未清理；
- 遍历过程中修改 set 引发集合大小变化错误。

### 7.3 Config、Compute 与测试

曾修复：

- Config 不接受浮点 MB，无法构造小容量溢出测试；
- HNArray 重构后 Chip 仍调用旧接口；
- VEX 增加 layer_num 后 Chip 未传入；
- 长上下文测试的 AB 实际过大，无法触发 HBM；
- Decode 输出 Token 多循环一次；
- 每轮 Decode 错误依赖 Prefill 的结束周期；
- Request 的 status 和 phase 混用；
- store 返回 False 后仍继续使用结果；
- 测试移到独立目录后的 Pylance 与 Ruff E402 导入问题；
- 原 Conda 环境包含绝对 prefix 和平台 Build，不适合迁移 Ubuntu 24.04。

## 8. 测试与验证

### 8.1 模块内置测试

[memory.py](hnlpu_sim/memory.py) 的内置测试覆盖：

- 默认320 MB Attention Buffer 填满；
- 小容量与 1、2、3、4、5、7-byte 非对齐分配；
- allocate → write → read → free；
- 容量不足后状态不变；
- HBM 大容量和小容量；
- HBM 共享 busy state 下的连续读写。

记录中报告：

~~~text
Large-capacity AttentionBuffer test: PASSED
Small-capacity AttentionBuffer test: PASSED
Large-capacity HBM test: PASSED
Small-capacity HBM test: PASSED
~~~

[kv_cache.py](hnlpu_sim/kv_cache.py) 的内置测试覆盖：

- Manager 全部公开方法；
- AB-only、HBM-only 和混合读取；
- AB 满后自动回退 HBM；
- AB 与 HBM 均满时拒绝存储；
- 单 Block 和整 Request 释放；
- 大小容量、多 Request、多 Layer 和一致性检查。

### 8.2 最终三份集成测试

测试文件：

- [test_request_chip.py](hnlpu_sim_test/test_request_chip.py)
- [test_request_chip_multilayer.py](hnlpu_sim_test/test_request_chip_multilayer.py)
- [test_request_chip_long_context.py](hnlpu_sim_test/test_request_chip_long_context.py)

最后一轮把每个 Transformer Layer 的流程细化为：

~~~text
RMSNorm
→ Q/K/V 并行
→ 历史 KV read，仅 Decode
→ Attention
→ Xo
→ Residual
→ RMSNorm
→ Router
→ Up/Gate 并行
→ SwiGLU
→ Down
→ Residual
→ KV store
~~~

测试建模约束：

- Q、K、V 在同一 request_cycle 发出，阶段结束取三者最晚完成周期；
- Up、Gate 同样并行；
- Decode 只读取历史 KV，当前 Token 的 K/V 来自本轮 projection；
- Prefill 不执行历史 KV read；
- 未单独执行 Softmax；
- Sampling 未加入每层流程；
- 未模拟 KV write 与计算重叠；
- MoE 固定选择本地第一个 Expert，只用于覆盖 expert-specific HNUnit 接口，不代表真实 Top-4 Router。

会话末尾报告：

~~~text
完整 hnlpu_sim_test：3 passed
Ruff：通过
git diff --check：通过
~~~

多层测试验证：

- 单 Chip、单 Request、Layer 0/1/2；
- KV 按 request_id 和 layer_id 隔离；
- 每层 Decode 只读本层历史 KV；
- 所有 Block ID 唯一。

长上下文测试使用人为缩小的 Buffer：

~~~text
attention_buffer_mb_per_chip = 0.064
attention_buffer_banks_per_chip = 32
attention_buffer_bank_kb = 2
~~~

场景为 Prefill 2 Token，再生成50 Token。会话末尾仍满足：

~~~text
首次 HBM spill：Token position 32
Attention Buffer usage：63,488 / 64,000 bytes
HBM usage：40,960 bytes
既出现 AB-only read，也出现 AB+HBM mixed read
最终生成50 Token，current position为52，Request finished
~~~

### 8.3 本次对当前工作树的复核

本次整理额外执行：

~~~text
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider hnlpu_sim_test
~~~

当前结果：

~~~text
3 passed in 0.21s
~~~

同时执行：

~~~text
ruff check hnlpu_sim hnlpu_sim_test
~~~

当前仓库级 Ruff 结果不是全绿，而是5项：

- hnlpu_sim/kv_cache.py：NumPy 未使用；
- hnlpu_sim/kv_cache.py：free_status 与 False 直接比较；
- hnlpu_sim/pipeline.py：Chip 和 Config 未使用；
- hnlpu_sim/system.py：Chip 未使用。

因此，“会话最后一轮报告 Ruff 通过”与“本次全目录 Ruff 检查失败”应分开理解：前者可能只检查了当时修改范围，后者是当前对 hnlpu_sim 与 hnlpu_sim_test 的仓库级复核。本次没有修改这些代码。

### 8.4 2026-08-24 AttentionBuffer 增量测试

新增 [test_attention_buffer.py](hnlpu_sim_test/test_attention_buffer.py)，把 AttentionBuffer 的碎片化分配从生产模块 main 区域之外单独纳入 pytest。最终7个定向测试覆盖：

- Stage 1 的 balanced single-group allocation 保持最高优先级；
- Stage 2 在 Bank usage 为 [4, 0, 0, 0]、bank size 为7 bytes、申请12 bytes 时，得到 {0: 3, 1: 4, 2: 4, 3: 1}，明确排除旧 greedy packing 的 {0: 3, 1: 7, 2: 2}；
- Stage 2 的 allocation 至少绕 Bank 一周，同一 Bank 在多轮中累计两个4-byte quantum；
- Stage 3 从非零 next group 开始跨多个 group，并在最后一个部分使用的 group 内保持 access-width round-robin 分布；
- 总容量真正不足时返回 False，且 usage、Bank/group usage、record 和 cursor 均不变化；
- 跨 group allocation 的 free 正确恢复所有容量账目；
- 跨 group allocation 的 allocate → write → read 保持读写总字节数正确，并自然覆盖多个 group 中的 Bank。

增量完成后的验证结果为：

~~~text
pytest -q hnlpu_sim_test/test_attention_buffer.py：7 passed
pytest -q hnlpu_sim_test：10 passed
memory.py 内置 AttentionBuffer/HBM 测试：PASSED
kv_cache.py 内置 KVcacheManager 测试：PASSED
ruff check hnlpu_sim/memory.py hnlpu_sim_test/test_attention_buffer.py：通过
git diff --check：通过
~~~

这里的10项 pytest 由7项 AttentionBuffer 定向测试和原有3项单 Chip 集成测试组成。仓库级 Ruff 的既有5项问题仍应按8.3节理解；本次只验证了实际修改的 memory.py 与 test_attention_buffer.py。

### 8.5 2026-08-27 至 28 Interconnect 增量测试

新增 [test_interconnect.py](hnlpu_sim_test/test_interconnect.py)，共18项 pytest case，覆盖：

- 4×4 topology 的16个 Chip；
- row 0、row 3、column 0 和 column 3 的线性 Chip ID；
- 同行和同列 direct link 存在，不同行且不同列的 link 不存在；
- 4×4 topology 恰好生成48条无向 link，且不存在反向重复 key；
- 所有 link_busy_until_cycle 初值均为0；
- 128 GB/s 到128,000,000,000 byte/s 的换算；
- 100 ns、1 GHz 到100 cycles 的换算，以及0 ns latency；
- collective_algorithms 六个字段的保存和外部字典拷贝隔离；
- 使用仓库 hnlpu_config.yaml 初始化 Interconnect；
- rows、cols、bandwidth、latency、clock 和 collective algorithm 的类型、范围与缺失字段校验。

增量完成后的验证结果为：

~~~text
pytest -q hnlpu_sim_test/test_interconnect.py：18 passed
pytest -q hnlpu_sim_test：28 passed
ruff check hnlpu_sim/interconnect.py hnlpu_sim/config.py hnlpu_sim_test/test_interconnect.py：通过
git diff --check：通过
~~~

仓库级 Ruff 仍有8.3节记录的5项既有问题，本次没有为通过全目录 lint 而修改无关模块。

### 8.6 2026-08-29 至 30 CommunicationTask 与 direct Reduce 测试

CommunicationTask 阶段新增41个展开后的 pytest case，覆盖七个字段、participants copy、六种 operation 的严格 root 语义、非法类型/范围、direction，以及明确不保存 algorithm 或 timing。完成时：

~~~text
pytest -q hnlpu_sim_test/test_interconnect.py：59 passed
pytest -q hnlpu_sim_test：69 passed
~~~

direct Reduce 阶段再新增16个展开后的 case，覆盖：

- 4-chip column Reduce 的三条 canonical links 和101-cycle service；
- required-link contention、无关 link 不阻塞，以及 serialization 向上取整；
- participant 范围、row/column/all direction 一致性；
- direct physical link 缺失时拒绝 multi-hop；
- tree 等未实现算法；
- single-participant no-op；
- 非法 task、operation、request_cycle；
- 所有异常路径保持 link state 不变。

最终验证结果：

~~~text
pytest -q hnlpu_sim_test/test_interconnect.py：75 passed
pytest -q hnlpu_sim_test：85 passed
ruff check hnlpu_sim/interconnect.py hnlpu_sim_test/test_interconnect.py：通过
git diff --check：通过
~~~

## 9. 当前文件状态（截至2026-08-31）

| 文件或目录 | 状态 |
|---|---|
| [hnlpu_config.yaml](hnlpu_config.yaml) | 已有模型、硬件、VEX、存储、互连和评估配置；Interconnect 使用128 GB/s/link、作为 simulator assumption 的100 ns，以及六项 direct 配置；当前只有 Reduce 消费其算法配置参与 timing |
| [performance_eval_simplify_codex.py](simplified_eval/performance_eval_simplify_codex.py) | 简化上下文长度性能评估 |
| [memory.py](hnlpu_sim/memory.py) | Memory、AttentionBuffer、HBM；AttentionBuffer 已支持三级分配、跨 group record、原子 plan/commit 和 access-width round-robin fallback |
| [kv_cache.py](hnlpu_sim/kv_cache.py) | KVcacheBlock、KVcacheManager 第一版及内置测试 |
| [request.py](hnlpu_sim/request.py) | Request 数据容器 |
| [config.py](hnlpu_sim/config.py) | YAML 解析、校验、单位换算和 overrides；已校验 Interconnect latency 与六项 collective algorithm 配置 |
| [compute.py](hnlpu_sim/compute.py) | ComputeTask、VEX、HNArray、HNUnit |
| [chip.py](hnlpu_sim/chip.py) | 单 Chip 资源聚合 |
| [pipeline.py](hnlpu_sim/pipeline.py) | 只有骨架，尚无真实调度 |
| [trace.py](hnlpu_sim/trace.py) | 尚未实现 |
| [interconnect.py](hnlpu_sim/interconnect.py) | 已完成 CommunicationTask、构造参数校验、单位换算、row/column groups、无向 link busy state 和 direct Reduce timing |
| [system.py](hnlpu_sim/system.py) | 尚未实现系统逻辑 |
| [hnlpu_sim_test](hnlpu_sim_test) | 三份单 Chip 集成测试、7项 AttentionBuffer 测试和75项 Interconnect 测试，当前共85项 pytest 通过 |
| [hnlpu-sim-ubuntu24.04.yml](hnlpu-sim-ubuntu24.04.yml) | 面向 Ubuntu 24.04 的便携 Conda 环境 |
| [hnlpu-sim-windows.yml](hnlpu-sim-windows.yml) | 当前 Windows 环境配置 |

Ubuntu 24.04 环境文件删除了原机器绝对 prefix、平台底层 Build 固定和大量传递依赖，保留 Python 3.11 与直接依赖。文件头已经包含创建和激活环境的命令；当前不含 CUDA 依赖。

## 10. 已知限制与高优先级风险

### 10.1 每 Chip 的 KV 容量口径尚未统一

对话中的论文映射分析曾按每个 Column group 负责2个 KV heads 估算：

~~~text
每Token、每Layer、每相关Chip：
2 × 2个KV heads × 64 head_dim × 2 bytes
= 512 bytes
~~~

而现有集成测试普遍使用完整8个 KV heads：

~~~text
2 × 8 × 64 × 2
= 2,048 bytes / Token / Layer
~~~

后者适合作为合成溢出测试，但不能直接代表论文多芯片映射下的单 Chip 容量压力。后续应建立唯一的 KV 容量计算入口，明确全局值、每层值和每 Chip 分片值。

### 10.2 VEX 按 Layer 独立可能高估并行度

对话在 8 月 21 日曾明确提醒：如果每颗 Chip 物理上只有一个共享 VEX，按 Layer 设置独立 busy state 会凭空增加资源。8 月 22 日根据用户明确要求实现了按 Layer 并行。

因此当前行为是模拟抽象，不应在未经论文或 RTL 核对前作为物理事实。

### 10.3 关键延迟仍未校准

以下均为假设或简化：

- HBM 6.4 TB/s；
- HBM 100 ns；
- Interconnect 当前采用100 ns 精确值；论文只报告 link latency 小于100 ns，因此该值是 simulator assumption；
- collective_algorithms 当前全部写为 direct，不代表论文确认的算法；其中只有 Reduce 已有 simulator timing，其余 operation 仍只是配置占位；
- HNArray 八类线性任务都为10 cycles；
- VEX 向量操作固定周期；
- Attention 使用 q_length × kv_length 的等价工作量；
- Expert 均匀分配与测试中的本地单 Expert；
- 解析吞吐模型的 2K Attention 时间占比。

当前实现更适合验证接口、时序关系和趋势，不适合直接给出最终硬件性能结论。

### 10.4 尚无统一 Pipeline 和事件调度

三份集成测试通过手工把上一步 finish_cycle 传给下一步。尚未统一实现：

- 任务依赖图；
- 事件队列；
- Pipeline stage/slot；
- 多 Request 调度；
- Continuous Batching；
- 访存与计算重叠；
- 跨层和跨 Chip 的真实并行关系。

### 10.5 其他限制

- HBM 双缓冲或 staging buffer 未实现；
- Interconnect 已完成 CommunicationTask 和 direct Reduce，但其他 collective、multi-hop、packet pipeline、full-duplex 方向独立状态、统一 execute、System 接入和完整 4×4 多芯片执行仍未实现；
- KV 主动迁移、淘汰、换入换出和自动 Block 拆分未实现；
- HBM 没有 Channel/Stack 级并发；
- Attention Buffer allocation 已能跨 Bank group，容量账目层面的 group/Bank 碎片不再导致提前 spill；但模型仍不保存真实地址或连续区间，也没有物理碎片整理；
- 混合 AB/HBM read 不具备严格的两阶段原子提交；
- free 不检查 allocation 是否仍有未完成访问；
- Request 缺少参数校验与封装的状态推进方法；
- Softmax 独立 timing、LM head/unembedding、完整 Sampling 未实现；
- Router、真实 Top-4 Expert 和跨 Chip Expert 通信未实现；
- consistency check 默认关闭时，公开可变状态被外部破坏后可能不能及时发现；
- 测试通过 sys.path.insert 和裸模块导入运行，尚未整理为正式 Python package；
- Memory 和 KV Cache 的大量测试仍放在生产模块的 main 区域；
- 解析评估器高度依赖校准，不能替代周期模拟器。

## 11. 建议的后续开发顺序

建议按以下顺序继续，避免过早扩展多个不稳定模块：

1. 建立统一的模型映射与 KV 容量公式，明确每 Token、每 Layer、每 Chip 的数据量。
2. 对照论文或 RTL 确认每 Chip 的 VEX 数量、Layer 映射和可并行资源。
3. 把论文值、派生值和人为假设在 Config 中明确分组，并为假设建立敏感性实验。
4. 实现最小 Pipeline/Simulator：事件队列、任务依赖、资源预约和完成事件。
5. 建立独立 DoubleBuffer/HBMStagingBuffer，由 Pipeline 模拟 HBM 加载与 VEX 计算重叠。
6. 实现 KV Block 自动拆分、主动迁移和淘汰；如需提高物理精度，再引入真实地址、连续区间和地址级碎片模型。
7. 在已有 CommunicationTask 和 direct Reduce 上，继续研究并逐项实现 broadcast、scatter、gather、all_reduce、all_gather，再统一设计 execute；随后接入16颗 Chip 与 System，验证 KV head、Token row 和 Expert 分布。
8. 实现 trace.py，支持真实 JSONL Trace 和合成长上下文负载。
9. 增加多请求、Batch、跨 Chip、双缓冲、异常原子性和论文校准点回归测试。
10. 将 hnlpu_sim 整理为正式 Python package，并把生产文件中的内置测试逐步迁移到独立测试目录。

## 12. 建议的 Trace 格式

对话中建议第一版使用 JSONL，每行一个请求：

~~~json
{"request_id": "request-0", "arrival_time_us": 0, "input_token_num": 32768, "output_token_num": 128}
{"request_id": "request-1", "arrival_time_us": 25, "input_token_num": 131072, "output_token_num": 64}
~~~

最低必需字段：

- request_id：唯一请求标识；
- arrival_time_us：相对于 Trace 开始时刻的到达时间；
- input_token_num：输入长度；
- output_token_num：Trace 中实际生成长度。

周期转换：

~~~text
arrival_cycle =
    arrival_time_us
    × clock_frequency_hz
    / 1,000,000
~~~

真实系统的 start/finish 可以保留为参考值，但不能写入 HNLPU 的模拟 start_cycle 和 finish_cycle；这两个值必须由模拟器根据资源竞争重新计算。

## 13. 后续协作时应保持的工作习惯

从对话中可以看出，用户偏好：

- 小步、局部、可验证地修改；
- 用户说只读分析时，不修改文件；
- 用户限定只改某个类或函数时，不顺带重构其他位置；
- 保留已有代码内容，只做注释或格式时，应保证可执行语义不变；
- 解释尽量通俗、简洁，并说明为什么需要周期或状态；
- 论文没有给出的参数必须标记为 assumed；
- 复杂一致性检查通过全局开关控制，测试时开启、性能模拟时关闭；
- 先跑通最小链路，再扩展到 Pipeline、多请求和多芯片。

这套偏好与当前项目阶段相匹配：优先保证职责清晰、状态可核对和测试可复现，再逐步提高物理精度。

## 14. 2026-08-24 至 25 增量同步：AttentionBuffer allocation

本节记录初次总结之后的两轮 AttentionBuffer 讨论和实现。实际代码提交发生在2026年8月24日，本文于8月25日同步。

### 14.1 第一轮：消除容量碎片导致的错误 HBM spill

第一轮修改对应提交 0d3cdb9，目标是在不重构整个 AttentionBuffer 的前提下解决两类错误拒绝：

1. 某个 group 总剩余容量足够，但 Bank 剩余容量不均匀，原 balanced striping 无法放入；
2. AB 总剩余容量足够，但任何单个 group 都无法完整容纳，原代码直接返回 False。

最终采用三级策略：

~~~text
Stage 1：balanced single-group allocation
Stage 2：single-group fallback
Stage 3：multi-group fallback
~~~

Stage 1 被完整保留为优先路径。Stage 2 选择第一个 aggregate remaining 足够的 group；Stage 3 在 AB 总容量足够时跨多个 group 规划。allocation record 从单值 bank_group 统一改为 bank_groups 字典，check_consistency 和 free_memory 做了最小必要适配，read/write 因始终读取 bank 映射而没有修改。

本轮特别确认了 allocation 的事务式语义：所有阶段先构建 planned_bank_allocations、planned_group_allocations 和计划 cursor；只有两级计划的总和都等于请求大小后才修改真实状态。AB 与 HBM 之间仍不拆分同一个 KV Block。

### 14.2 第二轮：fallback 从填满式 packing 改为 access-width round-robin

第二轮修改对应提交 ad2748d。讨论指出第一轮 Stage 2/3 虽然解决了容量可用性问题，但仍采用：

~~~python
allocated_to_bank = min(
    remaining_allocate_size,
    bank_remaining_size,
)
~~~

这种写法会先尽量填满当前 Bank，再访问下一个 Bank，使 fallback allocation 集中在少数 Bank 上，不符合既有 Bank 级并行模型的目标。

最终语义改为：

~~~python
allocated_to_bank = min(
    self.access_width_byte,
    bank_remaining_size,
    remaining_allocate_size,
)
~~~

Stage 2 在单个 group 内循环多轮；Stage 3 按 group round-robin，并在每个 group 内使用相同的多轮 Bank circular traversal。Bank 剩余少于 access width 时仍使用全部实际剩余字节。由于真实状态在规划阶段不能修改，计算 bank_remaining_size 时还必须减去 candidate_bank_allocations 中该 Bank 已经规划的字节；同一 Bank 多轮获得的数据则累加到 allocation record 的单个条目中。

为避免 circular traversal 死循环，每一完整 Bank round 都统计 allocated_this_round；如果仍有目标字节但本轮没有任何进展，则抛出 RuntimeError。cursor 始终依据最后一次实际 allocation 更新，而不是依据最后一次被检查但已满的 Bank 更新。

### 14.3 范围约束与当前结论

本次两轮代码修改仅涉及：

- [memory.py](hnlpu_sim/memory.py) 中 AttentionBuffer.allocate_memory、check_consistency 和 free_memory；第二轮只继续修改 allocate_memory 的 Stage 2/3；
- [test_attention_buffer.py](hnlpu_sim_test/test_attention_buffer.py) 的新增和断言更新。

没有修改 HBM、KVcacheManager、Chip、ComputeTask、HNArray、VEX、Interconnect、Pipeline 或 System。当前可以认为容量级 invariant 已收敛为：在参数合法、内部状态一致且 AB 总剩余容量不小于 allocation size 时，allocation 不会再因为 Bank/group 容量分布不均而返回 False；fallback 同时尽量保持 access-width 粒度的 Bank 间均匀分布。

## 15. 2026-08-27 至 28 增量同步：Interconnect 初始化

### 15.1 本轮范围与模块边界

本轮只完成 Interconnect 的构造和必要辅助初始化，主要修改：

- [interconnect.py](hnlpu_sim/interconnect.py)；
- [hnlpu_config.yaml](hnlpu_config.yaml)；
- [config.py](hnlpu_sim/config.py) 中读取新配置所需的最小校验；
- 新增 [test_interconnect.py](hnlpu_sim_test/test_interconnect.py)。

没有修改 Memory、KV Cache、Compute、HNArray、Chip、Pipeline 或 System。Interconnect 只模拟 Chip-to-Chip communication；Attention Buffer 与 HBM、Memory 与 Compute 以及其他片内数据传输继续由对应模块负责。

### 15.2 构造参数、派生量与配置归类

最终构造接口为：

~~~python
Interconnect(
    chip_grid_rows,
    chip_grid_cols,
    link_bandwidth_GBps,
    link_latency_ns,
    clock_frequency_hz,
    collective_algorithms,
)
~~~

其中 grid 尺寸、单 link 带宽和系统时钟属于硬件/系统配置输入；当前项目使用4×4、128 GB/s/link 和1 GHz。论文只给出 inter-chip link latency 小于100 ns，配置中的100 ns 是 simulator assumption，不是论文精确值。collective_algorithms 属于模拟器建模配置，六个 direct 仅为当前占位名称。

构造器保存原始单位，同时生成 num_chips、byte/s 带宽和使用 math.ceil 换算的 cycle latency。算法字典被复制，外部调用方后续修改原字典不会改变已创建对象。

### 15.3 Topology 与动态 link state

Chip 使用 row-major ID，row_groups 和 column_groups 使用 dict 保存。每个组内的所有 Chip pair 都有 direct link，因此 topology 是“同行全连接 + 同列全连接”，而不是16颗 Chip 全连接。

_build_links() 的三层核心过程是：遍历 row/column 两类 group，遍历每个 group 中的 source，再只遍历 source 之后的 destination。这样每一无向 pair 只产生一次；规范化 key `(min(src, dst), max(src, dst))` 又显式保证方向不会重复。对4×4 grid：

~~~text
row links    = 4 × C(4, 2) = 24
column links = 4 × C(4, 2) = 24
total links  = 48
~~~

静态 topology 与动态状态被分开理解：row_groups、column_groups 和 link key 集合描述连接关系；link_busy_until_cycle 的值描述每条物理 link 当前忙到哪个 cycle，并全部从0开始。当前 `(0, 1)` 同时代表0→1和1→0，两个方向共享同一资源状态；尚未建立 full-duplex 的方向独立状态。

### 15.4 Config 接入与校验

hnlpu_config.yaml 沿用 hnlpu.chip_grid_rows、hnlpu.chip_grid_cols 和 hnlpu.clock_GHz，避免在 interconnect section 建立重复配置源。interconnect section 保留已有 cxl_bandwidth_GBps_per_link 与 cxl_latency_ns，并加入六项 collective_algorithms。

Config 和 Interconnect 均校验 collective 字典的必需字段和值类型；Interconnect 还独立严格校验所有构造参数，包括拒绝 Python bool 被当作 int/float 接受。当前只校验算法名称是非空字符串，不预先限制算法集合。

### 15.5 测试结果与未实现边界

18项定向测试和完整28项 pytest 均通过，覆盖拓扑、link、单位换算、配置集成、字典拷贝和非法参数。实际修改范围的 Ruff 与 git diff --check 通过；全仓库仍有5项与本轮无关的既有 Ruff 问题。

本轮没有实现 all_reduce、all_gather、broadcast、reduce、scatter 或 gather，也没有为 direct、ring、tree 等名称加入 timing 公式。link_busy_until_cycle 目前只是已经初始化的动态状态容器，尚无通信方法更新它。下一阶段应先明确 collective 的参与 Chip group、通信步骤、每步数据量和 link 预约规则，再实现时延模型。

## 16. 2026-08-29 至 31 增量同步：CommunicationTask 与 direct Reduce

### 16.1 CommunicationTask 的职责与字段

CommunicationTask 参考 ComputeTask 的风格实现为纯数据容器，最终字段为：

- request_id：所属推理请求，必须非空且 hashable；
- operation：broadcast、reduce、scatter、gather、all_reduce、all_gather 之一；
- participants：参与通信的非负 Chip ID list，拒绝 bool、重复值和空 list，并复制保存；
- data_size_byte：以 Byte 表示的正整数通信数据量；
- source_chip、destination_chip：可选 root metadata，若提供则必须属于 participants；
- direction：可选的 row、column 或 all group 描述。

operation-specific 语义保持严格：broadcast/scatter 必须有 source 且没有 destination；reduce/gather 必须有 destination 且没有 source；all_reduce/all_gather 两者都必须为 None。CommunicationTask 不限制 Chip ID 小于16，也不判断 participants 是否真的属于某个 row/column/all group，这些 topology-specific check 留给 Interconnect。

Task 中没有 algorithm 和 request/start/finish/wait/service/total latency 字段。它只描述“做什么”，具体“怎么做”由 Interconnect.collective_algorithms 决定，request_cycle 则由执行通信的方法单独接收。

### 16.2 direct Reduce 的执行流程

Interconnect.reduce(task, request_cycle) 依次完成：

1. 校验 CommunicationTask 类型、reduce operation 和非负整数 request_cycle；
2. 校验 participants 与 destination 是否属于当前 Interconnect 的 Chip ID 范围；
3. 对 row/column direction 检查所有 participants 是否属于同一行/列，对 all 要求恰好包含全部 Chip；direction 为 None 时不做 group consistency check；
4. 读取 collective_algorithms["reduce"]，当前只接受 direct；
5. 为每个非 destination participant 构造 canonical physical link，并确认 link 实际存在；
6. 所有校验完成后计算 phase timing，最后只更新 required links 并返回 timing dict。

direct 的含义是每个 source 向 destination 直接发送一份完整 partial result。例如 participants 为 `[0, 4, 8, 12]`、destination 为8时，used_links 为：

~~~text
(0, 8)
(4, 8)
(8, 12)
~~~

若任意 source 与 destination 不同行且不同列，当前不会推导 multi-hop route，而是抛出 NotImplementedError。reduce algorithm 为 tree、ring 等未实现名称时也抛出 NotImplementedError。required links 会在任何状态修改之前完整确定，所以 validation、算法或 link 检查失败不会留下部分更新。

### 16.3 Reduce timing 与链路竞争

Reduce 中的 data_size_byte 表示每个 participant 上一份完整 partial result 的大小，不是 participant 数乘以该大小。单 link timing 为：

~~~text
bandwidth_byte_per_cycle = link_bandwidth_byte_per_s / clock_frequency_hz
serialization_cycles = max(1, ceil(data_size_byte / bandwidth_byte_per_cycle))
transfer_cycles = link_latency_cycles + serialization_cycles
~~~

当前128 GB/s、1 GHz 对应128 B/cycle；128-byte partial result 的 serialization 为1 cycle，加100-cycle fixed latency 后 service 为101 cycles。

不同 source 到 destination 的独立 physical links 在同一个 phase 中并行，而不是累加时间。整个 phase 等所有 required links 都空闲后统一开始：

~~~text
phase_start_cycle = max(request_cycle, required links 的最晚 busy cycle)
phase_finish_cycle = phase_start_cycle + transfer_cycles
~~~

随后所有 used_links 的 busy_until_cycle 统一更新为 phase_finish_cycle，无关 link 不受影响。若 participants 只有 destination，自然得到空 required_links，并在 request_cycle 原地完成，所有 timing 增量为0。

### 16.4 当前 simulator assumptions

以下均是当前模型抽象，不应表述成论文精确事实：

- Reduce arithmetic latency 视为0或与通信完全重叠；
- direct Reduce 的不同 physical links 可以并行；
- 一个 phase 等待所有 required links 可用后统一开始；
- link 在 fixed latency 加 serialization latency 的整个区间内保持 busy，尚无 packet pipeline；
- canonical 无向 link 的两个方向共享同一个 busy_until_cycle，尚未建模 full-duplex 独立状态；
- direct 缺少物理直连时不自动进行 multi-hop routing。

### 16.5 修改范围、测试与当前边界

两轮实现只修改 [interconnect.py](hnlpu_sim/interconnect.py) 和 [test_interconnect.py](hnlpu_sim_test/test_interconnect.py)。CommunicationTask 阶段只把 Interconnect.COLLECTIVE_OPERATIONS 接到共享 operation 常量，没有改变 Interconnect.__init__() 行为；Reduce 阶段没有改变 CommunicationTask.__init__() 或 Interconnect.__init__()。

最终验证为75项 Interconnect 定向测试和85项完整 pytest 全部通过，修改范围 Ruff 与 git diff --check 通过。当前只有 direct Reduce 拥有通信 timing；broadcast、scatter、gather、all_reduce、all_gather 和统一 Interconnect.execute() 仍明确未实现。
