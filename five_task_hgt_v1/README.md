# 5 任务 HGT：合并成对对抗后的关系分析

在干净信息集上，把成对对抗记录合并为一个事件，运行 5 任务 HGT（Event、Time、Position、Team、Player），逐项检验各关系对 5 个任务的作用，并分析 HGT 训练初期领先序列模型的原因。完整协议见 [PROTOCOL.md](PROTOCOL.md)，配置见 [configs/experiment.json](configs/experiment.json)。

## 目录

- `src/units.py`：合并单元、目标、名单与训练词表的重建。
- `src/batches.py`：窗口、在线控球阶段、异构图与序列输入。
- `src/models.py`：5 任务 HGT（旧 Representation 阶段的移植，HGT 层为 PyG `HGTConv` 参数加等价的快速前向）与序列对照、损失和 Player 梯度规则。
- `src/scoring.py`：逐样本统计、比赛级统计量、指标与配对 bootstrap。
- `src/fit.py`：单次训练；`src/queue_runs.py`：学习率搜索、锁定、正式运行与报告；`src/report.py`：汇总表、对比与学习曲线。
- `tests/test_five_task.py`：13 项验收（结果在 `reports/acceptance.json`）。
- `data/`、`runs/`、`reports/`：数据、运行与报告。`archive/interrupted_cpu_affinity_20261014/`：第一次启动因 DataLoader 子进程被绑到单核而停止的记录，未用于结果。

## 运行

```bash
cd /home/li/football/graph_prediction_research/five_task_hgt_v1
PY=/home/li/football/paper_1/Phase_1/.venv/bin/python
$PY -B src/units.py --config configs/experiment.json          # 已完成
$PY -B tests/test_five_task.py                                # 已通过
$PY -B src/queue_runs.py --config configs/experiment.json     # 后台运行中
cat reports/queue_status.json; tail -n 20 reports/queue.log
```

测试集不读取、不评估。
