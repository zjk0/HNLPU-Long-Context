# HNLPU 项目长会话上下文总结

> 整理日期：2026-08-23  
> 资料来源：指定 Codex JSONL 对话记录，以及对当前工作树的只读复核  
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

虽然文件被保存在 2026/07/01 目录中，但它不是只有 7 月 1 日的一轮对话，而是一条持续到 8 月 22 日、围绕同一仓库不断演进的长会话。原记录中的绝对路径使用 /media/kai，而当前工作树使用 /media/zjk；本文因此主要使用仓库相对路径。

项目主线可以概括为：

1. 先建立“性能指标随上下文长度变化”的解析评估脚本。
2. 随后将目标升级为不执行真实 Transformer 数值计算、但能够模拟容量、资源竞争和周期依赖的 HNLPU 性能模拟器。
3. 逐步完成 Memory、Attention Buffer、HBM、KV Cache、Request、Config、Compute 和 Chip 的第一版。
4. 最后通过单层、多层和长上下文三份单芯片集成测试，验证计算流程、KV 隔离以及 Attention Buffer 向 HBM 溢出。
5. Pipeline、双缓冲、Trace Loader、Interconnect、System、多请求与多芯片仍未真正实现。

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

Interconnect / System                尚未完成
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
- Interconnect/System：未来负责 4×4 多芯片拓扑、通信竞争和全局调度。

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
2. 一个 allocation 必须完整放在一个 group。
3. group 内按 4-byte word 条带化到 32 个 Bank。
4. 每个 group 保存 next offset，避免余数长期落在低编号 Bank。
5. group 总容量足够但某个目标 Bank 放不下时，继续搜索下一个 group。
6. 所有 group 都无法满足规则布局时返回 False，由 KVcacheManager 尝试 HBM。

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
    bank_group: 所属group
    bank:
        bank_id: 该allocation在此bank中的字节数
    ready_cycle: write完成后加入
~~~

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
- allocation 仍不能跨 Bank group。

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

## 9. 会话结束时的文件状态

| 文件或目录 | 状态 |
|---|---|
| [hnlpu_config.yaml](hnlpu_config.yaml) | 已有模型、硬件、VEX、存储、互连和评估配置 |
| [performance_eval_simplify_codex.py](simplified_eval/performance_eval_simplify_codex.py) | 简化上下文长度性能评估 |
| [memory.py](hnlpu_sim/memory.py) | Memory、AttentionBuffer、HBM 第一版及大量内置测试 |
| [kv_cache.py](hnlpu_sim/kv_cache.py) | KVcacheBlock、KVcacheManager 第一版及内置测试 |
| [request.py](hnlpu_sim/request.py) | Request 数据容器 |
| [config.py](hnlpu_sim/config.py) | YAML 解析、校验、单位换算和 overrides |
| [compute.py](hnlpu_sim/compute.py) | ComputeTask、VEX、HNArray、HNUnit |
| [chip.py](hnlpu_sim/chip.py) | 单 Chip 资源聚合 |
| [pipeline.py](hnlpu_sim/pipeline.py) | 只有骨架，尚无真实调度 |
| [trace.py](hnlpu_sim/trace.py) | 尚未实现 |
| [interconnect.py](hnlpu_sim/interconnect.py) | 尚未实现 |
| [system.py](hnlpu_sim/system.py) | 尚未实现系统逻辑 |
| [hnlpu_sim_test](hnlpu_sim_test) | 三份当前可通过的集成测试 |
| [hnlpu-sim-ubuntu24.04.yml](hnlpu-sim-ubuntu24.04.yml) | 面向 Ubuntu 24.04 的便携 Conda 环境 |

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
- Interconnect、CXL 竞争和 4×4 多芯片执行未实现；
- KV 主动迁移、淘汰、换入换出和自动 Block 拆分未实现；
- 一个 allocation 不能跨 Bank group，可能因内部碎片提前回退 HBM；
- HBM 没有 Channel/Stack 级并发；
- Attention Buffer 不保存真实地址，也没有碎片整理；
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
6. 实现 KV Block 自动拆分、主动迁移和淘汰，再研究 Bank group 内部碎片。
7. 实现 Interconnect 与16颗 Chip 的映射，验证 KV head、Token row 和 Expert 分布。
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
