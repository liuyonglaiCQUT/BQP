# 本机重训与历史 checkpoint 的区分

冻结日期：2026-10-18。

## 为什么需要本机重训

13 个原始 CP/RotatE checkpoint 不在本地；另有 FB15k-237/WN18RR 上 ComplEx/DistMult 的早期训练配置不完整。此前一次 CP 本机重训的验证 MRR、测试 MRR和逐查询排名均未与历史记录精确匹配。因此，本协议把新训练产物标记为 **local retraining**，不把它说成已恢复的原始权重或原始训练配置。

重训前，13 个历史 CP/RotatE 排名已复制到 `results_bqp/historical_architecture_ranks/`，并记录文件 SHA-256。主目录 `results_bqp/test_query_ranks/` 将在本机重训完成后更新为新权重对应的排名，再重跑相关分析；归档目录保留原服务器权重导出的排名用于比较。新的 ComplEx/DistMult 核心重训保存在独立的 `checkpoints/bqp_local_reproduction/`，不会覆盖历史核心权重。

## 固定的新训练配方

对 FB15k-237 与 WN18RR 的本地核心模型重训，模型结构沿用已保存的旧配置（ComplEx rank 200、DistMult rank 400、正态初始化标准差 0.001）。训练过程统一采用 reciprocal-query 负采样 logistic objective、Adam、每个正例 128 个均匀负实体、batch size 1,024、学习率 0.001、N3 系数 `1e-4`，最多 100 epoch，每 5 epoch 以 `all_known` filtered validation MRR 评估，patience 8，梯度范数裁剪 5。seed 为 7、11、19、23、31。

这些超参数是为可复现重跑而冻结的新配方，不表示原始核心 checkpoint 使用了相同设置。若验证性能明显偏离原结果，应如实报告；不得根据测试集调参。每次训练目录保存完整 `config.json`、训练数据文件哈希、验证轨迹、checkpoint 和 SHA-256 清单，并注明训练脚本哈希及其为本机重训。

CP/RotatE 的缺失运行沿用 `scripts/train_bqp_additional_architectures.py` 中已冻结的架构扩展配方，同样在配置中保存训练参数、数据划分哈希和本机重训来源。

## 执行顺序

1. 在主 checkpoint 根目录补跑缺失的 CP/RotatE。训练器自动跳过已经存在 `status=complete` 的运行：

   ```bash
   python scripts/train_bqp_additional_architectures.py --device cuda
   ```

   训练完成后，把 CP/RotatE 排名从当前 checkpoint 重新导出到正式矩阵（历史的 13 份排名已经归档）：

   ```bash
   python scripts/export_bqp_test_ranks.py \
     --dataset FB15k237 --dataset WN18RR \
     --model cp --model rotate --device cuda --force
   python scripts/audit_bqp_test_ranks.py
   ```

2. 将新增核心模型 checkpoint 写入隔离目录：

   ```bash
   python scripts/train_bqp_additional_architectures.py \
     --dataset FB15k237 --dataset WN18RR \
     --model complex --model distmult --device cuda \
     --output-root checkpoints/bqp_local_reproduction
   ```

3. 对新核心 checkpoint 生成独立排名，之后才运行对应核心统计：

   ```bash
   python scripts/export_bqp_test_ranks.py \
     --checkpoint-root checkpoints/bqp_local_reproduction \
     --output-root results_bqp/local_reproduction_test_ranks \
     --dataset FB15k237 --dataset WN18RR \
     --model complex --model distmult --device cuda
   ```

   NELL-995 的本地核心 checkpoint 使用主 checkpoint 根目录，因此单独导出到同一隔离排名目录：

   ```bash
   python scripts/export_bqp_test_ranks.py \
     --output-root results_bqp/local_reproduction_test_ranks \
     --dataset NELL995 --model complex --model distmult --device cuda
   ```

4. 把核心分析结果输出到独立报告和目录，不覆盖历史 checkpoint 分析：

   ```bash
   python scripts/run_bqp_core_experiment.py \
     --input-root results_bqp/local_reproduction_test_ranks \
     --output-root results_bqp/local_core_reproduction \
     --report docs/core_experiment_results_local_reproduction.md
   ```

全部本机训练和正式排名更新后，运行 `python scripts/audit_bqp_local_training_artifacts.py`。该审计检查新训练配置、checkpoint/trajectory/config 哈希、NELL 划分哈希、测试排名是否对应当前 checkpoint，以及归档历史排名的文件哈希。

## 解释边界

本地重训回答的是：在当前公开代码与记录配方下，BQP 分析能否独立重跑，以及结果对重新训练的敏感性如何。它不恢复被删除的早期配置，也不保证复现历史随机轨迹。历史分析与本地重训结果应并列比较，若排名或效应量改变，应把它报告为复现差异与稳健性证据，而不是悄悄替换历史结果。
