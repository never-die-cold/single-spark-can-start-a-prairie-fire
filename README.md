# Single Spark Can Start a Prairie Fire

探索多台低性能设备协作训练：用额外的局部计算，减少单节点参数存储和跨设备协商。项目工作名：**星火**。

当前版本包含 **NumPy 数学参考实现，以及四个真实子网络的单进程训练原型（M1）**。尚未实现真实多机传输、实测显存节省、训练加速或能耗优势；也没有确立学术首创性。

新增候选路线：**精确边界重放与按需选步**。节点复制一个小融合尾部，先上传最终候选；联合更新冲突时再上传少量中间步特征，协调端选定真实降损组合，节点通过确定性重算恢复参数。该路线不构建完整 Jacobian，代价是尾部副本与额外本地计算。设计、适用条件及已有工作的重合见 [docs/replay-design.md](docs/replay-design.md)。

## 当前研究对象

整体模型由多个设备上的子网络 `f_i` 和融合网络 `G` 组成。每台设备拥有不同的子网络参数。在一段固定数据、固定融合模型的训练阶段中：

1. 为各节点建立包含当前损失梯度信息的压缩响应矩阵。
2. 节点暂存本地候选更新，上传压缩响应及误差上界摘要。
3. 协调节点把压缩遗漏和 ReLU 切换误差计入联合损失上界。
4. 上界证明候选更新有利时才提交；信息不足时缩步、扩展响应空间或更新基准点。

候选贡献应集中在这个完整的接受/修复协议，不能把局部训练、随机种子、低秩压缩或零阶优化本身作为原创主张。

## 运行

Python 3.10+；依赖只有 NumPy。以下命令从仓库根目录运行：

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m prairie_fire.experiments --output results/latest.json
~~~

快速检查：

~~~bash
python -m prairie_fire.experiments --quick
~~~

完整实验同时包含早期的磁盘权重 XOR 零阶训练基线。该基线确实训练一个小网络，但**不能算作当前压缩响应协议的端到端训练结果**。

当前响应协议的完整小网络训练另行运行：

~~~bash
python -m prairie_fire.training --output results/m1-latest.json
~~~

该命令用种子 `17, 31, 47`，对同一数据、模型和初始参数运行普通反向传播、不压缩响应、压缩响应，并消融随机/梯度锚定空间及固定周期/按界刷新。每次候选包含 4 步真实子网络参数更新，拒绝保持原参数，共同提交后重新计算当前真实损失；每 8 轮单独训练融合模型并废弃旧契约。

目标损失预设为 `0.01`，每种配置最多 160 轮。JSON 分开记录安全检查与目标是否达成；未达目标的运行不能用来宣称达到同等效果时更省流量。短检查可用 `python -m prairie_fire.training --seeds 17 --rounds 12`。

实验说明、通信账本和当前继续判据见 [docs/m1-training.md](docs/m1-training.md)。

Windows 克隆、虚拟环境和运行说明见 [docs/windows.md](docs/windows.md)。

新路线与旧响应协议、普通反向传播的同任务对照：

~~~bash
python -m prairie_fire.replay --rounds 1500 --learning-rate .08 --fusion-learning-rate .2 --output results/replay-latest.json
~~~

所有方案使用相同的窗口、融合训练周期和初始参数。新增补发菜单、尾部和标签复制、重放与提交消息均纳入账本。实验中存在未达标和学习率敏感的情况，详见新路线说明。

保留原学习率的主对照中，新路线在两个种子达标，相对普通反向传播减少约 84% 和 90% 的逻辑通信；第三个种子未达标。中间步选择的消融未证明它比只发最终候选进一步省通信，节点尾部副本也增加了存储。完整正负结果见 [实验汇总](results/replay-assessment-2026-10-25.json)。

## 已放入的内容

| 文件 | 内容 |
| --- | --- |
| [docs/design.md](docs/design.md) | 假设、压缩反馈、非线性余项、联合接受条件及证明 |
| [docs/evolution.md](docs/evolution.md) | 早期方案到当前方案的演化，保留原创性主张的纠正 |
| [docs/related-work.md](docs/related-work.md) | 已识别先例、候选差异、尚未完成的查重 |
| [docs/roadmap.md](docs/roadmap.md) | 下一阶段实验及停止/继续判据 |
| [prairie_fire/networks.py](prairie_fire/networks.py) | 小型 ReLU 网络、完整 Jacobian 参考 oracle、非线性误差界 |
| [prairie_fire/contracts.py](prairie_fire/contracts.py) | 压缩响应、尾部界、节点摘要、联合接受、空间扩展和版本检查 |
| [prairie_fire/experiments.py](prairie_fire/experiments.py) | 固定种子的可复现检查及 JSON 报告 |
| [prairie_fire/training.py](prairie_fire/training.py) | 固定批次子网络训练、事务提交、契约修复、完整逻辑流量账本及六组对照 |
| [prairie_fire/baselines.py](prairie_fire/baselines.py) | 早期磁盘分块/标量同步 XOR 基线 |
| [tests/test_contracts.py](tests/test_contracts.py) | 性质测试、反例、版本和输入检查 |
| [tests/test_training.py](tests/test_training.py) | 批量反向传播、参数梯度保持、事务隔离和训练安全检查 |
| [results/reference-2026-10-25.json](results/reference-2026-10-25.json) | 本次完整运行产生的参考结果 |
| [results/m1-reference-2026-10-25.json](results/m1-reference-2026-10-25.json) | M1 多种子训练与消融结果；独立于原始数学基线 |

## 如何解读结果

- 梯度保持性质只针对**基准点**，不是说后续所有位置的梯度都无误差。
- 损失上界针对**固定的数据及标签、固定融合模型、可信节点摘要**，不是泛化保证或全局收敛证明。
- 代码使用普通 float64，不是经过向外舍入的区间算术验证器。
- 实现为了核对数学性质会显式构造完整 Jacobian；压缩矩阵字节数只是一个协议载荷估计，**不是当前程序的峰值内存测量**。
- `k+2` 个摘要数值按单样本计算；M1 逐样本建立契约，账本计入整个固定批次、元数据、构建/修复分发、所有重试和提交特征。
- 对整个试验的最终比较必须使用：达到同等目标效果所需的时间、全部网络字节数、峰值内存和总电量。

原始数学参考基线保留不变；M1 的新增结果单独保存。验证命令见上文，远程自动检查另行验证发布后的源码。

MIT License，保留仓库已有许可证。
