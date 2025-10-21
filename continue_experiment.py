import json
import os
import sys
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add the inferenceKit directory to the path
sys.path.append('/workspace/SSDP')
from inferenceKit.data_config import supported_dataset
from inferenceKit.model_config import create_llm, create_prm
from inferenceKit.models import InferenceConfig, DefaultInferenceModel
from accelerate import Accelerator

class RemainingDataset:
    """Custom dataset that only contains the remaining samples"""
    def __init__(self, full_dataset, remaining_indices):
        self.full_dataset = full_dataset
        self.remaining_indices = remaining_indices
        self.data = [full_dataset[i] for i in remaining_indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    print("=== Continuing GSM8K Qwen 1.5B DPTS Experiment ===")
    
    # Load completed samples
    resume_file = '/workspace/SSDP/results/gsm8k-qwen-1.5b-dpts/overnight-7-1/RESUME_last_max_0.json'
    with open(resume_file, 'r') as f:
        completed_samples = json.load(f)
    
    print(f"Found {len(completed_samples)} completed samples")
    
    # Load the original config
    config_file = '/workspace/SSDP/results/gsm8k-qwen-1.5b-dpts/overnight-7-1/config.json'
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Get the dataset
    dataset = supported_dataset['gsm8k']()
    total_samples = len(dataset)
    completed_indices = {sample['index'] for sample in completed_samples}
    
    # Find remaining samples
    remaining_samples = []
    for i in range(total_samples):
        if i + 1 not in completed_indices:  # +1 because dataset indices start from 1
            remaining_samples.append(i)
    
    print(f"Total samples in dataset: {total_samples}")
    print(f"Remaining samples to process: {len(remaining_samples)}")
    
    if len(remaining_samples) == 0:
        print("All samples already completed!")
        create_final_results(completed_samples)
        return
    
    print("Setting up models and continuing experiment...")
    
    # Initialize accelerator
    accelerator = Accelerator()
    accelerator.even_batches = False
    
    # Load models (same as original experiment)
    device_map = "auto"
    dtype = torch.bfloat16
    flash_attn = True
    
    # Create models
    reward_model_class, reward_model_path = create_prm('mistral_prm-7b', None)
    reward_model = reward_model_class(model_path=reward_model_path, device=accelerator.device, device_map=device_map, dtype=dtype, flash_attn=flash_attn)
    
    generation_model_class, generation_model_path = create_llm('qwen-1.5b', None, False)
    generation_model = generation_model_class(model_path=generation_model_path, device=accelerator.device, device_map=device_map, dtype=dtype, flash_attn=flash_attn)
    
    # Create inference model
    inference_config = InferenceConfig.from_json(config_file)
    model = DefaultInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device)
    model.eval()
    
    # Create custom dataset with remaining samples
    remaining_dataset = RemainingDataset(dataset, remaining_samples)
    dataloader = DataLoader(dataset=remaining_dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x[0])
    dataloader = accelerator.prepare(dataloader)
    
    print(f"Processing {len(remaining_samples)} remaining samples...")
    
    # Process remaining samples
    results = {}
    timing_info = {
        "total_inference_time": 0,
        "sample_times": [],
        "start_time": time.time()
    }
    
    try:
        for i, data in enumerate(tqdm(dataloader, desc="Processing remaining samples")):
            sample_start_time = time.time()
            input_text = data["input"]
            
            # Generate response using the model
            response_dict = model.generate(input_text)
            sample_end_time = time.time()
            
            timing_info["sample_times"].append({
                "index": data["index"],
                "time": sample_end_time - sample_start_time,
                "timestamp": sample_end_time
            })
            
            # Store results
            for k, response in response_dict.items():
                if k not in results:
                    results[k] = []
                res = data.copy()
                res["response"] = response
                results[k].append(res)
        
        timing_info["total_inference_time"] = time.time() - timing_info["start_time"]
        
        # Combine with completed samples
        print("Combining with completed samples...")
        final_results = {}
        for k in results.keys():
            # Load completed samples for this result type
            completed_file = f'/workspace/SSDP/results/gsm8k-qwen-1.5b-dpts/overnight-7-1/RESUME_{k}_0.json'
            if os.path.exists(completed_file):
                with open(completed_file, 'r') as f:
                    completed = json.load(f)
                final_results[k] = completed + results[k]
            else:
                final_results[k] = results[k]
        
        # Save final results
        base_dir = '/workspace/SSDP/results/gsm8k-qwen-1.5b-dpts/overnight-7-1'
        for k, res in final_results.items():
            result_file = f'{base_dir}/results-{k}.json'
            with open(result_file, 'w') as f:
                json.dump(res, f, indent=2)
            print(f"Saved {result_file} with {len(res)} samples")
        
        # Save timing info
        with open(f'{base_dir}/timing_info.json', 'w') as f:
            json.dump(timing_info, f, indent=2)
        
        print("Experiment continuation completed successfully!")
        print(f"Total samples processed: {len(final_results.get('last_max', []))}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Saving partial results...")
        # Save whatever we have so far
        create_final_results(completed_samples)

def create_final_results(completed_samples):
    """Fallback: create final results from completed samples only"""
    base_dir = '/workspace/SSDP/results/gsm8k-qwen-1.5b-dpts/overnight-7-1'
    
    result_types = ['last_max', 'last_vote', 'majority_vote', 'min_max', 'min_vote']
    
    for result_type in result_types:
        resume_file = f'{base_dir}/RESUME_{result_type}_0.json'
        result_file = f'{base_dir}/results-{result_type}.json'
        
        if os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                data = json.load(f)
            
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Created {result_file} with {len(data)} samples")

if __name__ == "__main__":
    main()
