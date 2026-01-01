from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re

def replace_llm_code(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'<llm-code>\s*', '```python\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*</llm-code>', '\n```', text, flags=re.IGNORECASE)
    return text


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
            # Chỉ lấy sample đúng
            if not ds[i].get('is_correct', False):
                continue
                
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
    subset_data = [{"question": ex["question"], "generated_solution": replace_llm_code(ex["generated_solution"])} for ex in subset_data]

    subset_ds = Dataset.from_list(subset_data)
    subset_ds.save_to_disk("data/subset_openmathinstruct_1_v2/256K")
    return subset_ds

def get_subset_dataset(target_gsm8k=10000, target_math=10000, seed=42, save_to_disk=True):
    # load dataset
    print("Loading OpenMathInstruct-1 dataset...")
    ds = load_dataset("nvidia/OpenMathInstruct-1", split='train')
    
    # grouping examples by dataset and question
    print("Grouping examples by dataset and question...")
    groups = {'gsm8k': defaultdict(list), 'math': defaultdict(list)}
    
    for i, ex in enumerate(tqdm(ds, desc='Iterating dataset')):
        # Chỉ xét các sample có is_correct == True
        if not ex.get('is_correct', False):
            continue
            
        dataset_name = ex.get('dataset')
        if dataset_name in ('gsm8k', 'math'):
            q = ex.get('question')
            groups[dataset_name][q].append(i)
    
    # create subsets
    print("Creating GSM8K subset with fair downsampling...")
    gsm8k_subset = get_fair_downsample_subset(groups['gsm8k'], target_gsm8k, seed=seed)
    
    print("Creating MATH subset with code filtering...")
    math_subset = get_any_code_filtering_subset(ds, groups['math'], target_math, seed=seed)
    
    # combine indices
    ds_indices = gsm8k_subset + math_subset
    print(f"Total samples: {len(ds_indices)}")
    
    # create and save subset
    if save_to_disk:
        subset_ds = save_subset(ds, ds_indices)
        print(f"Subset saved to data/subset_openmathinstruct_1")
    else:
        subset_data = [ds[i] for i in ds_indices]
        subset_data = [{"question": ex["question"], "generated_solution": replace_llm_code(ex["generated_solution"])} 
                      for ex in subset_data]
        subset_ds = Dataset.from_list(subset_data)
    
    return subset_ds

if __name__ == "__main__":
    print("testing get_subset_dataset function...")
    print("=" * 60)
    
    subset_ds = get_subset_dataset(
        target_gsm8k=256000,
        target_math=256000,
        seed=42,
        save_to_disk=True
    )
    
    print("\n" + "=" * 60)
    print(f"Successfully created subset dataset")
    print(f"Total samples: {len(subset_ds)}")
    print(f"\nFirst sample:")
    print(f"Question: {subset_ds[0]['question'][:100]}...")
    print(f"Solution: {subset_ds[0]['generated_solution'][:100]}...")