from common.instrument import require_l40s, hardware_info
import torch
info = require_l40s()
print("require_l40s passed")
for k, v in info.items():
    print(f"  {k:16s} {v}")
print(f"  n_gpus_visible   {torch.cuda.device_count()}")
p = torch.cuda.get_device_properties(0)
print(f"  gpu_mem_GB       {p.total_memory/1e9:.1f}")
