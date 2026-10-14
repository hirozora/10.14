# 运行说明

## 当前执行

使用 Python：`/home/li/football/paper_1/Phase_1/.venv/bin/python`。

已完成训练和验证数据重建、三个选择种子的全部 C2 配对及阶段 A 验收。训练队列运行在 GPU 0–3，每张卡一个独立进程；队列 PID 写入 [queue_pid.json](reports/queue_pid.json)。

执行顺序为：

1. 12 次学习率预实验：2 个任务 × D00/D11 × 3 个学习率，每次 24 轮。
2. 每个任务使用选定学习率，从头训练 D00 的三个 seed，按正式停止规则运行。
3. 用对抗修正验证主指标的三种子样本标准差 ×3 锁定阈值。
4. 执行 B/C/D 探索与预登记三种子确认，复用同配置 D00 基线；所有 C2 规则进入确认。
5. 生成阶段报告、24 项主要比较、q>0 次要配对分析及跨模型采用判据。

外部 HGT/GRU 的实现和运行留到采用判据产生之后；当前队列不包含阶段 E，也没有测试评估入口。

## 查看进度

工作目录为 `/home/li/football/graph_prediction_research`。

```bash
cat reports/queue_status.json
tail -n 20 reports/queue.log
```

每次运行在 `runs/<run_id>/` 保存 `status.json`、`curve.jsonl`、`config.json`、`sample_order.npy`、最佳与末轮检查点。完整验证每四轮执行；完成前不能据早期领先判断采用。每次运行的标准输出单独写入 `reports/<run_id>.log`。

## 显式命令

以下是实际流程入口，已有数据或队列存在时会立即失败；不要对已运行目录重复执行。

```bash
/home/li/football/paper_1/Phase_1/.venv/bin/python src/data.py --config configs/experiments.json --step rebuild
/home/li/football/paper_1/Phase_1/.venv/bin/python src/data.py --config configs/experiments.json --step selections
/home/li/football/paper_1/Phase_1/.venv/bin/python src/acceptance.py --config configs/experiments.json
/home/li/football/paper_1/Phase_1/.venv/bin/python src/run_experiments.py --config configs/experiments.json
```

队列遇到子进程错误时停止派发新任务，等待已运行任务结束后报错；不会自动恢复、换数据或改配置。验收失败时不能训练；阈值未锁定时不能启动其余正式条件。

## 实现约定

- 空原始子类型编码为未知 `−1`；padding 通过独立 mask 排除。
- 球场尺度使用 105×68 米；百分比坐标先除以 100，再换算到锚点球队的进攻方向。
- 对抗配对沿用同 period、异队、相邻 Duel、起点镜像误差小于 0.011 的规则。换算后的重合验收使用相同容差。
- 控球关闭事件归入它关闭的阶段；其后的事件从未知 owner 开始。争抢不单凭执行球队变化切换 owner。
- 时间偏置 MLP 输入为 Δn、Δt、Δt 有效性和三种关系标记，中间宽度 16，输出 4 个头的分数。D00/D10 仅将关系标记置零。
- 连续时间消融关闭连续时间及有效性，保留事件距离。未知身份编码 0，不建立同身份关系。
- 输入侧始终重新计算所选记录的身份与控球编号，模型不编码未选记录。选择器的候选元数据单独使用。
- 参数报告同时区分总参数、有梯度张量中的元素数，以及训练期间实际出现非零梯度的元素数；不会把关系关闭后的失活列算作相同有效容量。
- 学习率搜索取每个参照在 24 轮内的最低逐记录验证监控值，再平均两个参照。并列取较小学习率。
- bootstrap 种子固定为 20261014；按比赛抽样的同一组权重用于全部主要比较。区间条件于已训练模型。

已有数据和旧实验目录只读。所有新产物保存在本目录。
