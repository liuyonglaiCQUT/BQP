# BQP 正式测试集逐查询结果协议

## 目的

第一阶段只生成冻结的正式分析输入，不在测试集上探索假设、调参或选择 checkpoint。输出同时保留
方向查询级表示与源三元组配对表示，后续可直接比较“把 2N 条查询独立重采样”与“把 N 个三元组
作为簇重采样”的不确定性估计。

## 实验矩阵与选择规则

最终排名矩阵是分层扩展设计，共 50 个运行：FB15k-237 与 WN18RR 的四架构矩阵40个运行，加上
NELL-995 的ComplEx/DistMult核心矩阵10个运行。核心关系控制分析包含三个数据集的ComplEx与DistMult，
共30个运行。NELL-995 使用本项目由完整图派生的实体覆盖划分：

| 数据集 | 模型 | seeds |
|---|---|---|
| FB15k-237 | ComplEx、DistMult | 7、11、19、23、31 |
| WN18RR | ComplEx、DistMult | 7、11、19、23、31 |
| NELL-995（派生实体覆盖划分） | ComplEx、DistMult | 7、11、19、23、31 |
| FB15k-237、WN18RR | CP、RotatE | 7、11、19、23、31 |

NELL-995 的目标总体限定为派生 test split 的 3,992 个源事实及其两个方向；该 split 保留每个实体
至少一条训练事实，不声称复现官方 NELL-995 的事实划分。三个数据集上的不确定性区间均条件于各自
固定图、固定 split 和固定 checkpoint，不外推为图间、split 间或训练种子总体的不确定性。

每个 checkpoint 在旧训练运行中按验证集 micro MRR 选择，验证过滤协议为 `all_known`。迁移时删除了
优化器状态，只保留推理所需权重、配置、选择 epoch、验证 MRR 和原始文件哈希。正式导出不重新选择
checkpoint。

## 测试排名协议

- 每个测试三元组 `(h, r, t)` 生成两个查询：尾预测 `(h, r, ?)` 与头预测 `(t, r^-1, ?)`；
- filtered ranking 删除 train、valid、test 中同一查询的其他已知真答案；
- 目标答案不被删除；
- 并列候选使用 average rank：`1 + #greater + 0.5 × (#equal - 1)`；
- 每个模型运行的全局 MRR 等于全部头、尾方向 reciprocal rank 的 micro average。

测试指标用于描述和后续冻结分析，不参与 checkpoint 选择。

## 文件与字段

每个运行输出四个文件。

### `test_query_ranks.npz`

长度为 `2N` 的字段：

- `source_triple_ids`：原始测试三元组编号；前 N 个和后 N 个均为 `0..N-1`；
- `directions`：`0=tail`，`1=head`；
- `original_heads`、`base_relations`、`original_tails`：原始三元组；
- `query_heads`、`query_relations`、`target_entities`：统一转换成尾预测形式的实际查询；
- `filtered_ranks`、`reciprocal_ranks`：过滤排名及其倒数。

标量字段包括 `selected_epoch`、`seed`、`num_entities`、`num_relations`、
`validation_filter_protocol` 和 `test_filter_protocol`。

### `test_queries.csv.gz`

一行对应一个方向查询，同时提供整数 ID 和原始实体、关系名称。它适合查询级描述统计；进行 BQP
推断时必须按 `source_triple_id` 聚类。

### `test_triple_pairs.csv.gz`

一行对应一个原始测试三元组，`tail_rank` 与 `head_rank` 保持配对。这是簇 bootstrap、头尾协方差、
design effect 和有效样本量分析的首选输入。

### `test_metrics.json`

保存 checkpoint 选择信息、全局/方向/关系指标、过滤与并列协议、运行时间、文件哈希，以及从旧运行
迁移后在当前硬件重算 MRR 的漂移。由于不同 GPU 和 PyTorch 矩阵乘法可使极少数近并列候选交换顺序，
复现门槛设为绝对 MRR 差 `1e-4`；本矩阵最大差为 `2.18e-5`。

## 已完成审计

执行：

```bash
python scripts/audit_bqp_test_ranks.py
```

审计会验证：

1. 50 个运行身份唯一且均有输出；
2. NPZ 每个向量的长度和方向顺序正确；
3. 原始三元组、尾查询和互逆头查询与数据集逐项一致；
4. rank 合法且 reciprocal rank、MRR、MR、Hits@K 可以从原始数组复算；
5. 查询 CSV 有 `2N` 行，配对 CSV 有 `N` 行；
6. 三个数据文件的 SHA-256 与报告一致；
7. 当前硬件重算 MRR 落在迁移复现门槛内。

最终预期累计 511,920 个“模型运行 × 源三元组”观测与 1,023,840 个方向查询；其中 ComplEx/DistMult
核心子集对应 275,920 个源三元组观测与 551,840 个方向查询。具体审计状态和哈希保存在
`results_bqp/test_query_ranks/audit.json`。

