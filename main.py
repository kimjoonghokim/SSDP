import argparse
import os
import datetime
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from inferenceKit import utils
from inferenceKit.model_config import create_llm, create_prm, create_embedding_model
from inferenceKit.data_config import supported_dataset
from inferenceKit.inference import inference
from inferenceKit.models import InferenceConfig, DefaultInferenceModel, VLLMInferenceModel

import torch.nn.functional as F
import random
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser()
    SUPPRESS = argparse.SUPPRESS
    
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='Json Config File specifying arguments')
    
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model',type=str, required=True)
    parser.add_argument('--model_path',type=str, default=None)
    parser.add_argument('--reward_model', type=str, default=None)
    parser.add_argument('--reward_model_path', type=str, default=None)
    
    # Embedding model arguments for clustering
    parser.add_argument('--embedding_model', type=str, default=None, help='Embedding model for node clustering')
    parser.add_argument('--embedding_model_path', type=str, default=None, help='Path to embedding model')
    parser.add_argument('--enable_clustering', action='store_true', default=False, help='Enable embedding-based node clustering')
    parser.add_argument('--clustering_threshold', type=float, default=None, help='Similarity threshold for clustering (0.0-1.0)')
    parser.add_argument('--clustering_method', type=str, default=None, help='Clustering method: cosine_similarity, kmeans, hierarchical')
    
    parser.add_argument('--vllm', action='store_true', default=False, help='Enable vllm')
    
    parser.add_argument('--cot_method', type=str, default=SUPPRESS)
    parser.add_argument('--step_method', type=str, default=SUPPRESS)
    parser.add_argument('--voting_method', type=str, default=SUPPRESS)

    parser.add_argument('--max_step_time', type=int, default=SUPPRESS)
    
    # Safe optimization arguments (no threading, no parameter changes)
    parser.add_argument('--fast_mode', action='store_true', default=False, help='Use fast mode with reduced parameters')

    parser.add_argument('--work-dir',type=str, default='./outputs')
    parser.add_argument('--exp-name', type=str, default=None)
    
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--flash-attn', action='store_true', default=False, help='Enable Flash Attention')
    parser.add_argument('--shard', action='store_true', default=False, help='Big Model Sharded Inference')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug Mode')
    parser.add_argument('--resume', type=str, default=None, help='Resume work-dir Experiment')
    
    args = parser.parse_args()
    return args


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_all()

    args = parse_args()
    args = vars(args)
    accelerator = Accelerator()
    accelerator.even_batches = False
    logger = utils.get_logger('MAIN')

    device_map = "auto" if args.pop("shard") or args["vllm"] else None
    dtype = getattr(torch, args.pop("dtype"))
    flash_attn = args.pop("flash_attn")
    
    # Extract optimization arguments
    fast_mode = args.pop("fast_mode", False)
    
    # Check if resuming before processing config
    resume_path = args.pop("resume", None)
    is_resume = resume_path is not None

    config_file = args.pop("config", None)
    
    # Handle resume case
    if is_resume:
        # Load config from the resume directory
        config_path = os.path.join(resume_path, "config.json")
        if os.path.exists(config_path):
            inference_config = InferenceConfig.from_json(config_path)
        else:
            inference_config = InferenceConfig()
    else:
        inference_config = InferenceConfig.from_json(config_file) if config_file else InferenceConfig()
        # Only update config with command line arguments that were actually provided (not defaults)
        # Filter out clustering-related defaults that would override config file settings
        clustering_defaults = {
            'embedding_model': None,
            'embedding_model_path': None, 
            'enable_clustering': False,
            'clustering_threshold': None,
            'clustering_method': None
        }
        provided_args = {k: v for k, v in args.items() if k not in clustering_defaults or v != clustering_defaults[k]}
        utils.update_single_config(inference_config.config, copy=False, **provided_args)
    
    args = inference_config.config
    
    # Only access cot_method if it exists (not when resuming)
    if hasattr(args, 'cot_method'):
        args.cot_method = args.cot_method.lower()
    
    # Apply ONLY safe optimizations that don't break tensor dimensions
    if fast_mode:
        print("Applying safe fast mode optimizations...")
        # Only reduce max_step_time and max_rollout, keep tree_width/num_beams the same
        if hasattr(args, 'max_step_time') and args.max_step_time > 60:
            print(f"Reducing max_step_time from {args.max_step_time} to 60")
            args.max_step_time = 60
        
        if hasattr(args, 'max_rollout') and args.max_rollout > 10:
            print(f"Reducing max_rollout from {args.max_rollout} to 10")
            args.max_rollout = 10
    
    print(args)
        
    reward_model = None
    if args.reward_model is not None:
        reward_model, args.reward_model_path = create_prm(args.reward_model, args.reward_model_path)
        reward_model = reward_model(model_path=args.reward_model_path, device=accelerator.device, device_map=device_map, dtype=dtype, flash_attn=flash_attn)
    
    # Initialize embedding model for clustering
    embedding_model = None
    if hasattr(args, 'embedding_model') and args.embedding_model is not None and hasattr(args, 'enable_clustering') and args.enable_clustering:
        embedding_model_path = getattr(args, 'embedding_model_path', None)
        embedding_model_class, embedding_model_path = create_embedding_model(args.embedding_model, embedding_model_path)
        embedding_model = embedding_model_class(model_path=embedding_model_path, device=accelerator.device)
        print(f"Initialized embedding model: {args.embedding_model} with path: {embedding_model_path}")
    
    generation_model, args.model_path = create_llm(args.model, args.model_path, args.vllm)
    generation_model = generation_model(model_path=args.model_path, device=accelerator.device, device_map=device_map, dtype=dtype, flash_attn=flash_attn)

    if args.vllm:
        model = VLLMInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device, embedding_model=embedding_model)
    else:
        model = DefaultInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device, embedding_model=embedding_model)
    model.eval()
    
    print("initialize dataloader")
    # initialize dataloader
    dataset = supported_dataset[args.data]()
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, collate_fn=lambda x:x[0])

    # distributed
    dataloader = accelerator.prepare(dataloader)
    
    print("initialize output directory")
    # initialize output directory
    final_inference_config = model.inference_config
    output_dir = None
    if accelerator.is_main_process:
        if is_resume:
            output_dir = resume_path
            # TODO: Implement resume functionality
            print(f"Resuming experiment from {resume_path}")
            print("Note: Resume functionality is not fully implemented")
            # resume_file = utils.get_resume_file(resume_path)
            # resume_results = utils.collect_resume_results(resume_file)
        else:
            work_dir = utils.default_work_dir(args, final_inference_config)
            output_dir = utils.get_outdir(work_dir)
    output_dir = broadcast_object_list([output_dir], from_process=0)[0]
    
    if accelerator.is_main_process:
        final_inference_config.to_json(os.path.join(output_dir, "config.json"))
    
    print("inference")
    start_time = time.time()
    results = inference(model, dataloader, accelerator, output_dir)
    end_time = time.time()
    peak_memory = torch.cuda.max_memory_allocated()
    
    total_questions = 0
    evaluation_results = {}
    for k, res in results.items():
        all_results = accelerator.gather_for_metrics(res, True)
        
        if accelerator.is_main_process:
            all_results = sorted(all_results, key=lambda x: x["index"])
            utils.dump(all_results, os.path.join(output_dir, f"results-{k}.json"))
            
            # Capture evaluation results
            eval_result = dataset.evaluate_results(all_results)
            evaluation_results[k] = eval_result
            total_questions = len(all_results)
    
    # Calculate average time per question
    avg_time_per_question = (end_time - start_time) / total_questions if total_questions > 0 else 0
    
    print("**** finish evaluate, config: ****")
    print(args)
    print(f"Overall inference time: {end_time - start_time:.3f} s, peak memory: {peak_memory / 1024 ** 3:.2f} GB")
    print(f"Total questions processed: {total_questions}, Average time per question: {avg_time_per_question:.3f} s")
    
    # Print clustering metrics if available
    clustering_file = os.path.join(output_dir, "clustering_metrics.json")
    if os.path.exists(clustering_file):
        try:
            with open(clustering_file, 'r') as f:
                clustering_metrics = json.load(f)
            
            if clustering_metrics.get("total_samples", 0) > 0:
                print(f"\n" + "="*60)
                print(f"FINAL CLUSTERING EFFICIENCY SUMMARY")
                print(f"="*60)
                print(f"📊 Dataset: {clustering_metrics['total_samples']} samples processed")
                total_samples = clustering_metrics.get('total_samples', 1)
                avg_explorations = clustering_metrics['total_node_explorations'] / total_samples
                avg_nodes = clustering_metrics['total_all_nodes'] / total_samples
                avg_pruned = clustering_metrics.get('total_nodes_pruned', 0) / total_samples
                avg_clustering_applied = clustering_metrics.get('clustering_applied_count', 0) / total_samples
                
                print(f"🌳 Nodes: {clustering_metrics['total_node_explorations']:,} node explorations → {clustering_metrics['total_all_nodes']:,} total nodes in tree")
                print(f"💾 Clustering: {clustering_metrics.get('total_nodes_pruned', 0):,} nodes pruned ({clustering_metrics.get('clustering_applied_count', 0)} times applied)")
                print(f"📊 Averages per sample: {avg_explorations:.1f} explorations, {avg_nodes:.1f} nodes, {avg_pruned:.1f} pruned, {avg_clustering_applied:.1f} clustering applications")
                print(f"="*60)
        except Exception as e:
            print(f"Note: Could not load clustering metrics: {e}")
    
    # Save inference metrics and evaluation results to files
    if accelerator.is_main_process:
        inference_metrics = {
            "inference_time_seconds": end_time - start_time,
            "peak_memory_gb": peak_memory / 1024 ** 3,
            "total_questions": total_questions,
            "avg_time_per_question_seconds": avg_time_per_question,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args)
        }
        utils.dump(inference_metrics, os.path.join(output_dir, "inference_metrics.json"))
        
        # Save evaluation results for each voting method
        utils.dump(evaluation_results, os.path.join(output_dir, "evaluation_results.json"))

if __name__ == '__main__':
    main()