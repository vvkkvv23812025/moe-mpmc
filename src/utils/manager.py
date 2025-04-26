import glob
import math
import os
import queue
from threading import Thread

import torch
import time
import GPUtil

from model.predictor import SimpleLSTMClassifierSparseAttention
from model.soda_moe import SwitchTransformersSparseMLPOffloading


def load_oracle_hash(ckpt_path, topk=1, ratio=None, split="validation"):
    pathlist = glob.glob(f"{ckpt_path}/activation_{split}_large-*.pt")
    if len(pathlist) == 0:
        pathlist = [f"{ckpt_path}/activation_{split}_large.pt"]
    if ratio is not None:
        assert len(pathlist) > 1, "load ratio should not be set unless there is more than one checkpoint batch"
        # Use floor to avoid out of index range
        pathlist = pathlist[math.floor(len(pathlist) * ratio) :]

    raw_dict_list = [torch.load(path) for path in pathlist]

    hash_table = {key: torch.tensor([]) for key in raw_dict_list[0].keys()}
    for dictionary in raw_dict_list:
        for i, key in enumerate(dictionary.keys(), 1):
            # Gather all tensors under the same key
            tensor_list = [dictionary[key][k] for k in list(dictionary[key].keys())[1:]]
            # Concatenate along the first dimension
            data_tensor = torch.cat(tensor_list, dim=0)
            # Find the argmax for each line
            topk_indices = torch.topk(data_tensor, k=topk, dim=-1).indices
            hash_table[key] = torch.cat((hash_table[key], topk_indices), dim=0).int()
    return hash_table


def move_to_device(module, device):
    """
    Move all parameters and buffers in the module except the target experts to the given device recursively.
    """
    for param_name, param in module.named_parameters(recurse=False):
        module._parameters[param_name] = param.to(device)
    for buffer_name, buffer in module.named_buffers(recurse=False):
        module._buffers[buffer_name] = buffer.to(device)
    for submodule_name, submodule in module.named_children():
        if not isinstance(submodule, SwitchTransformersSparseMLPOffloading):
            move_to_device(submodule, device)
        else:
            pass


class SODAManager:
    def __init__(self, model, batched_loader, n_experts, topk=1, predictor_model=SimpleLSTMClassifierSparseAttention()):
        self.emb = model.switch_transformers.get_input_embeddings().to("cuda")
        self.loader_iter = iter(batched_loader)
        self.predictor = predictor_model.to("cuda")
        self.expert_module = {}
        self.topk = topk
        self.module_keys = [n.replace("-", ".") for n in self.predictor.y_keys]

        for n, m in model.named_modules():
            if n + ".router.classifier" in self.module_keys:
                self.expert_module[n + ".router.classifier"] = m
        self.expert_status = {k: {"cpu": set(range(n_experts)), "gpu": set()} for k in self.expert_module.keys()}

    def gen_hash(self, *, soft_target: bool = False):
        """
        Build the hash table for the *next* batch.
    
        Assumptions (per user spec & Algorithm 2):
            • self.predictor() already emits the replicated expert IDs needed
              for the upcoming batch—one ID per (token, layer) position.
            • If soft_target=True it returns a tuple (values, indices) where
              `values` are the soft targets (probabilities) and `indices`
              are the replicated expert IDs.
            • self.module_keys is an L-length list mapping MoE layers to keys.
        Returns
        -------
        hash_table : dict  {layer-key ➜ tensor or (values, indices)}
        expert_lists : dict {layer-key ➜ set(active expert IDs)}
        """
        try:
            batch = next(self.loader_iter)          # X_{i+1} in Algorithm 2
        except StopIteration:
            return None, None
    
        with torch.no_grad():
            emb_out = self.emb(batch["input_ids"].to("cuda"))
            pred_out = self.predictor(emb_out)
        if soft_target:
            values, indices = pred_out
            values, indices = values.cpu(), indices.cpu()
        else:
            indices = pred_out.cpu()
    
        hash_table   = {}
        expert_lists = {}
    
        for l, key in enumerate(self.module_keys):
            if soft_target:
                layer_val = values[:, :, l]         # [B, T]
                layer_idx = indices[:, :, l]        # [B, T]
                hash_table[key] = (layer_val, layer_idx)
            else:
                layer_idx = indices[:, :, l]        # [B, T]
                hash_table[key] = layer_idx
    
            expert_lists[key] = set(torch.unique(layer_idx).numpy().tolist())
    
        return hash_table, expert_lists


    def move_experts(self, expert_lists):
        for name, module in self.expert_module.items():
            to_gpu = self.expert_status[name]["cpu"] & expert_lists[name]
            to_cpu = self.expert_status[name]["gpu"] - expert_lists[name]

            self.expert_status[name]["cpu"] |= to_cpu
            self.expert_status[name]["gpu"] -= to_cpu
            self.expert_status[name]["gpu"] |= to_gpu
            self.expert_status[name]["cpu"] -= to_gpu
            # Move back to cpu first to avoid OOM
            for id in to_cpu:
                module.experts[f"expert_{int(id)}"].to("cpu")
            for id in to_gpu:
                module.experts[f"expert_{int(id)}"].to("cuda")


class SODAThread(Thread):
    def __init__(self, manager: SODAManager, q: queue.Queue):
        super().__init__()
        self.manager = manager
        self.q = q

    def run(self):
        hash_table = {}
        while hash_table is not None:
            hash_table, expert_lists = self.manager.gen_hash()
            self.q.put((hash_table, expert_lists), block=True)


class GPUMon(Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        # Global list to store GPU utilizations
        self.avg_util = 0
        self.util_steps = 0
        self.monitoring = False

    def run(self):
        """Function to periodically monitor and store GPU utilization"""
        while self.monitoring:
            GPUs = GPUtil.getGPUs()
            total_utilization = sum([gpu.load for gpu in GPUs]) / len(GPUs)
            self.avg_util = (self.avg_util * self.util_steps + total_utilization) / (self.util_steps + 1)
            self.util_steps += 1
            time.sleep(0.1)  # Check every 0.1 seconds

    def start(self):
        self.monitoring = True
        return super().start()

    def join(self, timeout=None):
        self.monitoring = False
        return super().join(timeout)


class monitor_gputil:
    def __init__(self, interval=0.1):
        self.mon_thread = GPUMon(interval)

    def __enter__(self):
        self.mon_thread.start()
        return self.mon_thread

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mon_thread.join()
        print(f"Average GPU Utilization over time during inference: {self.mon_thread.avg_util*100:.2f}%")
