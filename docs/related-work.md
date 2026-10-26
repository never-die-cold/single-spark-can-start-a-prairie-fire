# 已识别先例与候选差异

记录日期：2026-10-25。下列链接来自本次讨论中核对的作者页面/原始论文。本文件不是系统综述或完整的新颖性检索。

## 已有先例

| 工作 | 时间 | 与本项目相关的已有机制 |
| --- | --- | --- |
| [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) | 2017 | 共同随机数、标量通信、大量并行工作节点 |
| [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333) | 2023 | 正负扰动前向、零阶估计、种子重建扰动、低内存微调 |
| [FedKSeed](https://arxiv.org/abs/2312.06353) | 2023 预印本；ICML 2024 | 多客户端通过种子和标量梯度进行联邦全参数微调 |
| [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning](https://arxiv.org/abs/2402.11592) | 2024 | 分块下降、混合训练、梯度稀疏化 |
| [ZO2](https://arxiv.org/abs/2503.12668) | 2025 | CPU/GPU 参数卸载与双前向计算组织 |
| [DistZO2](https://arxiv.org/abs/2507.03211) | 2025 | 零阶训练的多设备并行与参数搬运 |
| [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343) | 2016 预印本；ICML 2017 | 用局部代理反馈解耦模块训练 |
| [CoeF-SFL](https://arxiv.org/pdf/2609.34360) | 2026 预印本，论文标注 under review | 复用远端梯度、曲率补偿、Jacobian 敏感度、随机投影 |

不同工作的实验范围不同：强化学习、语言模型微调、多 GPU 计算等均不能直接作为“任意旧设备从零预训练大模型”的证据。

## 与当前候选最接近的部分

CoeF-J 的式 (10)–(12) 已经考虑输出空间曲率上界和 `JᵀJ` 敏感度，实际以随机投影降低 Jacobian 下发成本。其说明区分了响应线性变化时成立的界，以及其他情形的近似分析。

因此不能将“下发局部目标”“Jacobian 压缩”“局部多步更新”中的任意一个单独当作本项目的新贡献。

当前拟进一步核查的组合为：

1. 包含基准损失梯度方向的**共享输出空间**，加上显式空间外响应界。
2. 允许跨越 ReLU 激活边界的、按输出影响加权的**非线性余项预算**。
3. 对不同参数子网络的共同更新，使用低维摘要计算**联合接受条件**。
4. 按尾部/非线性界的失败原因选择扩展空间或更新基准点，并完整计入修复成本。

这些都是候选差异，不是已经证明的优先权。正交投影、范数界、局部二次模型、majorization 和误差触发通信本身都是已有数学/系统工具。

## 建立新颖性还需要什么

- 对低秩曲率、合成梯度、分裂学习、非线性验证和事件触发通信做更系统的查重。
- 检查已有一般框架是否已经包含本协议作为特例，不能只比较关键词。
- 做移除梯度锚、移除尾部界、移除切换界、固定周期刷新等消融。
- 证明或测量修复成本何时能被复用收益抵消。
- 与相同模型、数据和精度目标的基线比较，而非仅展示几个新符号。

当前结论：已有明确先例；候选协议值得继续检验，但**尚未确立学术首创性**。

## 新的精确边界重放路线

2026-10-25 补充核对：[FedBCD](https://arxiv.org/abs/1912.11187) 已包含多次本地更新；[EF-VFL](https://arxiv.org/html/2406.14420v2) 第 3 节明确讨论在客户端执行融合模型前后向和多步本地更新；[Speculative Approximations for Terascale Distributed Gradient Descent Optimization](https://faculty.ucmerced.edu/frusu/Papers/Conference/2015-danac-gd.pdf) 已研究多个候选的推测式试算与选择。

因此，复制融合尾部、缓存、减少同步和候选选择不能单独视为新贡献。新原型的候选差别是聚合边界、冲突时增量请求前缀特征菜单，以及联合选步后的确定性重放事务；详细机制与限制见 [replay-design.md](replay-design.md)。尚未完成完整新颖性审查或对这些论文的实现排名。
