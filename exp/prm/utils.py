"""
Utility functions for PRM
"""

import subprocess
import time


def stats():
    """Check GPU stats"""
    u = int(subprocess.check_output(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", 
        shell=True
    ))
    used = int(subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", 
        shell=True
    ))
    total = int(subprocess.check_output(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", 
        shell=True
    ))
    return u, used, total


def wait_for_gpu(min_free_mb=20000):
    """Wait for GPU to be available"""
    while True:
        u, used, total = stats()
        free = total - used
        print(f"GPU util={u}%, free={free}MB")
        if u < 10 and free >= min_free_mb:
            break
        time.sleep(10)


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison"""
    ans = ans.replace(',', '').replace(' ', '').lower()
    # Try to parse as number
    try:
        return str(float(ans))
    except:
        return ans
