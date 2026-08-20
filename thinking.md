随Context Length增加，HNLPU的首要瓶颈是什么？VEX / Attention Buffer / HBM / Interconnect / Pipeline？

AB容量在什么Context Length开始不足？

HBM访问增加之后，有多少延迟能够被Pipeline隐藏？

为什么论文256K基本没有明显stall，而512K出现约10.7%的stall？

不同KV placement策略是否会改变exposed HBM stall？

Prefetch / double buffering是否能够进一步隐藏长上下文HBM访问？

长上下文是否会改变六阶段pipeline各stage之间的负载平衡？

其实我对HNLPU长上下文的研究的主要担心的点在于，我自己写的这个模拟器是否可信，能不能正确反映HNLPU的某些行为或者某些数据。因为论文中有很多细节，很多参数都没有给出，都需要自己去假设。

多种bank选择策略的实验。