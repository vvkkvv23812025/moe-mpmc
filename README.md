# MoE-MPMC
Fast Mixture-of-Experts Inference via Predictive Prefetching and Expert Replication  
(Code Artifact for SC 25)
---


MoE-MPMC is a system that accelerates inference for Mixture-of-Experts (MoE) language-model layers.  
The key idea is to **predict** which experts will be active in the *next* batch and then **replicate** those hot experts on the GPU so that concurrent tokens never wait in line.

* **Predictive prefetching** is handled by a lightweight 10-layer Simple Recurrent Unit (SRU) that achieves roughly 99.4% accuracy in matching the soft-max router.
* **MPMC replication** (Multiple-Producer Multiple-Consumer) deep-copies each predicted expert just enough times to saturate GPU compute, subject to a hard cap that avoids memory overflow.
* The method is applied at inference and prediction is learnt during finetuning.

---

## Paper contributions (in brief)

1. **SRU-based next-batch expert prediction** reduces routing overhead compared with the LSTM used in prior work of SiDA-MoE.  
2. **Data-aware expert replication** removes token queuing and lifts GPU utilisation from single-digit percentages in baseline SwitchTransformers to above 90 % on A100-80 GB.  
3. **End-to-end speed-up** of up to three times latency on Switch-Base-128 while maintaining 90–95 % of baseline accuracy.  

---
