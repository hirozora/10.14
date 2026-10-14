# 历史终点追加实验

> **2026-10-14 复核：本实验结果无效（泄露）。** 增益全部来自锚点是成对对抗第二条记录的 13,021 个样本（三 seed 准确率 +6.0/+7.8/+7.4 pp，位置 −5.6 到 −6.3 m）；其余 70,839 个样本变化约为 0。原因是同一次对抗的另一条记录的终点就是对抗结果：它与下一事件起点 1 m 以内的比例为 91.9%，而锚点自身起点只有 1.9%。后续改为把成对记录合并为一个事件，见 `../five_task_hgt_v1/`。

固定以当前已完成三种子中 Event 准确率最高的 `C2_current_phase_control` 为基础，只追加一个 ETP 条件、三个 seed。保留历史终点，屏蔽锚点终点数值和有效性；助攻／关键传球仍是 Raw10 Pass。新增终点投影层带来 192 个参数，其他公共参数按同 seed 配对初始化。

- [追加协议](PROTOCOL.md)
- [完整配置](configs/experiments.json)
- [终点数据审计](reports/endpoint_data_audit.json)
- [验收结果](reports/acceptance.json)及[日志](reports/acceptance.log)：8 项通过。
- [追加队列状态](reports/queue_status.json)
- [追加队列日志](reports/queue.log)

后台入口是 `src/after_current.py`。它等待父队列进程退出并确认父状态为 `complete` 后，才在 GPU 0、1、2 启动三个 seed。不会修改父队列，不会重新选择组或学习率；已有输出则立即失败。错误写入 `reports/queue_error.json`。

每个 seed 按原协议训练，最多 120 轮；不是追加一个 epoch，也不是续训旧检查点。运行文件保存在本目录 `runs/`。完成后生成 `reports/comparison.json`，与原组逐 seed 配对，报告三个评价口径及 q>0 子集。

这是一轮根据已有验证结果选组后的探索性追加，不加入原 24 项主检验族。测试集继续关闭。

```bash
cat reports/queue_status.json
tail -n 20 reports/queue.log
```
