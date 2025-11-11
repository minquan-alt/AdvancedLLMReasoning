from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re

def get_fair_downsample_subset(q2indices, target, seed=42):
    ran = random.Random(seed)
    questions = list(q2indices.keys())
    ran.shuffle(questions)
    result = []
    q_deques = {}

    for q in questions:
        ls_indices = q2indices[q][:]
        ran.shuffle(ls_indices)
        q_deques[q] = deque(ls_indices)

    q_cycle = deque(questions)

    while q_cycle and len(result) < target:
        q = q_cycle.popleft()
        dq = q_deques[q]
        if dq:
            result.append(dq.popleft())
        if dq:
            q_cycle.append(q)
  
    return result

def get_any_code_filtering_subset(ds, q2indices, target, seed=42):
    result = []
    for q, indices in tqdm(q2indices.items(), desc='processing any code filtering'):
        code_indices = []
        text_indices = []
        for i in indices:
            em = ds[i].get('error_message')
            code_used = (em != '<not_executed>')
            if code_used:
                code_indices.append(i)
            else:
                text_indices.append(i)
        if code_indices:
            result.extend(code_indices)
        else:
            result.extend(text_indices)
        
    ran = random.Random(seed)
    ran.shuffle(result)
    return result[:target]

def save_subset(ds, ds_indices):
  subset_data = [ds[i] for i in ds_indices]
  subset_data = [{"question": ex["question"], "generated_solution": ex["generated_solution"]} for ex in subset_data]

  subset_ds = Dataset.from_list(subset_data)
  subset_ds.save_to_disk("data/subset_openmathinstruct_1")
  return subset_ds