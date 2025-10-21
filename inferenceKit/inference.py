import sys
import os
import traceback
import time
import json
from os import PathLike
from tqdm import tqdm
from typing import Union, Dict, List
import glob

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from inferenceKit import utils
from .utils import InferenceConfig
from .models import BaseInferenceModel


def inference(
        model: BaseInferenceModel, 
        dataloader: DataLoader, 
        accelerator: Accelerator,
        output_dir: str,
        **kwargs,
    ) -> List[Dict]:
    
    prev_tmpl = os.path.join(output_dir,'{}_{}_{}_PREV.json')
    ind, num = accelerator.process_index, accelerator.num_processes
    
    results = {}
    clustering_metrics_aggregate = {
        "total_node_explorations": 0,
        "total_nodes_pruned": 0,
        "total_terminated_nodes": 0,
        "total_all_nodes": 0,
        "clustering_applied_count": 0,
        "total_samples": 0
    }
    timing_info = {
        "total_inference_time": 0,
        "sample_times": [],
        "start_time": time.time()
    }
    
    try:
        for i, data in enumerate(tqdm(dataloader)):
            sample_start_time = time.time()
            input = data["input"]
            response_dict = model.generate(input, **kwargs)
            sample_end_time = time.time()

            timing_info["sample_times"].append({
                "index": data["index"],
                "time": sample_end_time - sample_start_time,
                "timestamp": sample_end_time
            })

            res = data
            for k, response in response_dict.items():
                if k not in results:
                    results[k] = []
                res["response"] = response
                
                # Collect clustering metrics from model instance if available
                if hasattr(model, '_last_clustering_metrics') and model._last_clustering_metrics:
                    metrics = model._last_clustering_metrics
                    print(f"[AGGREGATION DEBUG] Sample {data['index']}: pruned={metrics.get('total_nodes_pruned', 0)}, applied={metrics.get('clustering_applied_count', 0)}")
                    clustering_metrics_aggregate["total_node_explorations"] += metrics.get("total_node_explorations", 0)
                    clustering_metrics_aggregate["total_nodes_pruned"] += metrics.get("total_nodes_pruned", 0)
                    clustering_metrics_aggregate["total_terminated_nodes"] += metrics.get("total_terminated_nodes", 0)
                    clustering_metrics_aggregate["total_all_nodes"] += metrics.get("total_all_nodes", 0)
                    clustering_metrics_aggregate["clustering_applied_count"] += metrics.get("clustering_applied_count", 0)
                    clustering_metrics_aggregate["total_samples"] += 1
                    print(f"[AGGREGATION DEBUG] Running totals: pruned={clustering_metrics_aggregate['total_nodes_pruned']}, applied={clustering_metrics_aggregate['clustering_applied_count']}")
                    # Clear the metrics after collecting
                    model._last_clustering_metrics = None
                
                results[k].append(res)
            
            for k in results.keys():
                utils.dump(results[k], prev_tmpl.format(ind, num, k))
    
    except (Exception, KeyboardInterrupt) as e:
        for k, res in results.items():
            all_results = accelerator.gather_for_metrics(res, True)
            if accelerator.is_main_process:
                all_results = sorted(all_results, key=lambda x: x["index"])
                resume_tmpl = os.path.join(output_dir, 'RESUME_{}_{}.json')
                i=0
                while os.path.exists(resume_tmpl.format(k, i)):
                    i += 1
                utils.dump(all_results, resume_tmpl.format(k, i))
                
                for file in glob.iglob(prev_tmpl.format("*", "*", k)):
                    os.remove(file)
        traceback.print_exc()
        sys.exit()
    
    timing_info["total_inference_time"] = time.time() - timing_info["start_time"]
    
    # Calculate simplified clustering metrics
    if clustering_metrics_aggregate["total_samples"] > 0:
        clustering_metrics_aggregate["avg_node_explorations_per_sample"] = clustering_metrics_aggregate["total_node_explorations"] / clustering_metrics_aggregate["total_samples"]
        clustering_metrics_aggregate["avg_nodes_pruned_per_sample"] = clustering_metrics_aggregate["total_nodes_pruned"] / clustering_metrics_aggregate["total_samples"]
        clustering_metrics_aggregate["avg_terminated_nodes_per_sample"] = clustering_metrics_aggregate["total_terminated_nodes"] / clustering_metrics_aggregate["total_samples"]
        clustering_metrics_aggregate["avg_all_nodes_per_sample"] = clustering_metrics_aggregate["total_all_nodes"] / clustering_metrics_aggregate["total_samples"]
        clustering_metrics_aggregate["avg_clustering_applied_per_sample"] = clustering_metrics_aggregate["clustering_applied_count"] / clustering_metrics_aggregate["total_samples"]
    
    # Save timing information and clustering metrics
    if accelerator.is_main_process:
        timing_file = os.path.join(output_dir, "timing_info.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_info, f, indent=2)
        
        # Save clustering metrics
        clustering_file = os.path.join(output_dir, "clustering_metrics.json")
        with open(clustering_file, 'w') as f:
            json.dump(clustering_metrics_aggregate, f, indent=2)
        
        print(f"Timing info saved to {timing_file}")
        print(f"Clustering metrics saved to {clustering_file}")
        print(f"Total inference time: {timing_info['total_inference_time']:.3f}s")
        if timing_info["sample_times"]:
            avg_time = sum(t['time'] for t in timing_info['sample_times']) / len(timing_info['sample_times'])
            print(f"Average sample time: {avg_time:.3f}s")
        
        # Print brief clustering metrics summary (detailed metrics shown in final results)
        if clustering_metrics_aggregate["total_samples"] > 0:
            print(f"Clustering: {clustering_metrics_aggregate['total_node_explorations']:,} explorations → {clustering_metrics_aggregate['total_all_nodes']:,} final ({clustering_metrics_aggregate.get('nodes_saved_by_clustering', 0):,} pruned)")
        
    return results