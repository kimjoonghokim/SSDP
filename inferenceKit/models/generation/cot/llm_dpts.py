import time
import math
import random
from functools import partial
from typing import List, Set
from types import MethodType

import torch

from .base import Searcher, Node
from .clustering import cluster_parallel_nodes
from transformers.cache_utils import DynamicCache
from transformers import StoppingCriteria
import heapq
import copy
import numpy as np

from torch.profiler import profile, ProfilerActivity

EXPLOIT_STATUS = 10000
EXPLORE_STATUS = 10001

def getMemoryUsage():
    allocated_memory = torch.cuda.memory_allocated() 
    total_memory = torch.cuda.get_device_properties(0).total_memory  
    return allocated_memory / total_memory

class SeqNode(Node):
    cnt = 0
    
    def __init__(self, seq, len=0, parent=None, value_score=0.2, reward_score=0.2):
        """ Sequence-Level Monte-Carlo Tree Node

        Args:
            input_ids (torch.LongTensor): (batch_size, seq_length)
            parent (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(parent)
        self.seq = seq
        self.len = len
        self.value_score = value_score
        self.reward_score = reward_score
        self.this_seq = seq[ :, -len:]

        self.Q = 0.0
        self.N = 1

        self.terminated = False
        
    def __repr__(self):
        return f"SeqNode_{self.id}"

    def get_all_len(self):
        cur_node = self
        total_len = 0
        while cur_node:
            total_len += cur_node.len
            cur_node = cur_node.parent
        return total_len

    def update_kv(self, past_key_values: DynamicCache, n_seq_len: int=0, subtree_idx: int=0, kv_len=None): 

        self.cntprev_kv_len = kv_len
        for layer_idx in range(len(past_key_values)):
            key, value = past_key_values[layer_idx]
            key = key[:, :, :kv_len, :]
            value = value[:, :, :kv_len, :]
            self.past_key_values.update(key[subtree_idx:subtree_idx+1, : , -(n_seq_len):, :], \
                                        value[subtree_idx:subtree_idx+1, : , -(n_seq_len):, :], \
                                        layer_idx=layer_idx)
            
        if len(self.parent.past_key_values.key_cache) == 0: 
            self.parent.cntprev_kv_len = kv_len - n_seq_len

            for layer_idx in range(len(past_key_values)):
                key, value = past_key_values[layer_idx]
                self.parent.past_key_values.update(key[:1, : , :self.parent.cntprev_kv_len, :], \
                                            value[:1, : , :self.parent.cntprev_kv_len, :], \
                                            layer_idx=layer_idx)
      
    def get_all_kvlen(self):
        tot_len = 0
        cur_node = self
        while cur_node is not None:
            if hasattr(cur_node, 'past_key_values'):
                tot_len += cur_node.past_key_values.key_cache[0].shape[-2]
            cur_node = cur_node.parent  

        return tot_len

    def get_path_len(self):
        tot_path_len = 0
        cur_node = self
        while cur_node is not None:
            tot_path_len += 1   
            cur_node = cur_node.parent  

        return tot_path_len

    def clear_kv(self):
        if hasattr(self, 'past_key_values'):
            del self.past_key_values 
            torch.cuda.empty_cache()

class CoT_DPTS_Searcher(Searcher):
    def __init__(self, max_rollout, max_step_time, expand_fn, reward_fn, terminated_fn, \
                exploration_weight=(1/math.sqrt(2)), weight_scheduler=None, importance_fn=None, \
                inference_config=None, generation_config=None):
        super().__init__()
        self.max_rollout = max_rollout
        self.max_step_time = max_step_time
        
        self.expand_fn = expand_fn
        self.reward_fn = reward_fn
        self.terminated_fn = terminated_fn
        self.importance_fn = importance_fn
        
        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler

        self.lambda_es = inference_config.config.lambda_es
        self.lambda_ds = inference_config.config.lambda_ds
        self.t_star = inference_config.config.t_star
        self.p = 0.5

        self.inference_config = inference_config
        self.generation_config = generation_config
        
        self.terminated_node = set()  
        self.finished_node = set()
        self.all_nodes = set()
        self.expanded_nodes = set()  # Track which nodes have been expanded (needed for rollout logic)

        # Simplified node tracking metrics
        self.total_node_explorations = 0  # Total number of node explorations (including re-explorations)
        self.total_nodes_pruned = 0       # Nodes removed by clustering
        self.clustering_applied_count = 0

        self.max_parallel_num = inference_config.llm_config.num_beams
        self.init_memory_ratio = getMemoryUsage()

    def process(self, inputs):
        cur_node = self._get_root_node(inputs)
        best_action = self._search(cur_node)
        return self.finalize()
    
    def get_clustering_metrics(self):
        """Get simplified clustering metrics"""
        total_terminated_nodes = len(self.terminated_node)
        total_all_nodes = len(self.all_nodes)
        
        return {
            "total_node_explorations": self.total_node_explorations,
            "total_nodes_pruned": self.total_nodes_pruned,
            "total_terminated_nodes": total_terminated_nodes,
            "total_all_nodes": total_all_nodes,
            "clustering_applied_count": self.clustering_applied_count,
            "nodes_saved_by_clustering": self.total_nodes_pruned if self.clustering_applied_count > 0 else 0
        }
    
    def finalize(self):
        
        if len(self.terminated_node) == 0:
            return [self.root_node.seq], [[0]]

        seq_list = []
        score_list = []
        for node in self.terminated_node:
            score = []
            cur_node = node
            while cur_node != None:
                score.append(cur_node.reward_score)
                cur_node = cur_node.parent
            score.reverse()
            
            seq_list.append(node.seq)
            score_list.append(score)
        
        return seq_list, score_list

    def cnt_paral_num(self, rollouts):
        if rollouts == 0:
            memory_ratio = getMemoryUsage()
            paral_num = (1)//memory_ratio  
            paral_num = int(min(paral_num, self.max_parallel_num))
        else:
            peak_ratio = getMemoryUsage()
            memory_ratio = peak_ratio - self.init_memory_ratio
            paral_num = (1-self.init_memory_ratio)//memory_ratio 
            paral_num = int(min(paral_num, self.max_parallel_num))
        return paral_num

    def _search(self, node):
        
        total_time = 0
        rollouts = 0
        reward_list = []
        reward_for_all_explores = []

        self.all_nodes.add(node)
        selected_paral_node = {node: EXPLOIT_STATUS} 

        while total_time <= self.max_step_time: 
            start_time = time.time()

            selected_paral_node = self._select(node, paral_num=self.cnt_paral_num(rollouts), selected_paral_node=selected_paral_node)

            if (not isinstance(selected_paral_node, SeqNode)) and len(selected_paral_node) == 0: break
            if ((len(reward_list) >= self.t_star)) and max([x.reward_score for x in list(selected_paral_node)]) < max(reward_list): break

            path, end_node, rollout_path = self._rollout(selected_paral_node)
            
            for t_path in rollout_path:
                first_node = t_path[0]
                last_node = t_path[-1]
                
                theta_early_stop = np.mean(reward_for_all_explores) * self.lambda_es
                theta_deep_seek = np.mean(reward_for_all_explores) * self.lambda_es

                status = selected_paral_node.pop(first_node, None)
                if (status == EXPLOIT_STATUS and (len(reward_for_all_explores) <= 1 or last_node.reward_score > theta_early_stop)) \
                    or (status == EXPLORE_STATUS and last_node.reward_score >= theta_deep_seek):
                    selected_paral_node.update({last_node: EXPLOIT_STATUS})
                    reward_for_all_explores.append(last_node.reward_score)
            
            for e_nd, end_path in path.items():
                first_node = end_path[0]
                selected_paral_node.pop(first_node, None)

            selected_paral_node = {key: value for key, value in selected_paral_node.items() if value == EXPLOIT_STATUS}

            if isinstance(end_node, SeqNode):
                reward = self._reward(path, end_node)
                reward_list.append(reward)
                self._backpropagate(end_node, reward)
            else: 
                for t_end_node in end_node:
                    reward = self._reward(path[t_end_node], t_end_node)
                    reward_list.append(reward)
                    self._backpropagate(t_end_node, reward)

            end_time = time.time()
            total_time += end_time - start_time
            rollouts += 1
        
        return max(node.children, key=self._select_best_action) if node.children else node
    
    def _get_best_path(self):
        if len(self.terminated_node) == 0:
            return [[]]
        best_node = max(self.terminated_node, key=lambda i:i.reward_score)
        return best_node


    def _select(self, node: SeqNode, paral_num=1, selected_paral_node=None):
        if len(self.all_nodes) - len(self.finished_node) == 0:
            return set()
        
        topk = paral_num - len(selected_paral_node)
        loop_count = 0
        while (isinstance(node, SeqNode) and (node in self.expanded_nodes or node in self.terminated_node) ) or \
              (isinstance(node, (set, list)) and any((i in self.expanded_nodes or i in self.terminated_node) for i in node)): 

            loop_count += 1
            if loop_count > 100:  return set()
            
            node = self._best_node(topk=topk)
        
        exploit_node_num = int(paral_num * self.p) 
        exploit_node_num -= len(selected_paral_node) 
        if exploit_node_num > 0 and not isinstance(node, SeqNode):
            candidate_exploit_node = heapq.nlargest(exploit_node_num, node, key=lambda x: x.reward_score)
        else: 
            candidate_exploit_node = set()
        
        if isinstance(node, SeqNode): node = set((node, ))

        for nd in node:
            if nd in candidate_exploit_node:
                selected_paral_node.update({nd: EXPLOIT_STATUS})
            else:
                selected_paral_node.update({nd: EXPLORE_STATUS})

        return selected_paral_node

    def _rollout(self, node):
        ret_end_node_path = {} 
        has_expanded = False
    
        cur_node = list(node.keys())
        rollout_path = [[cur_node[_]] for _ in range(len(cur_node))]
        if len(cur_node) == 1: cur_node = cur_node[0]
        
        def terminate_fn(cur_node, rollout_path, has_expanded):
            if isinstance(cur_node, SeqNode): 
                if_terminated = self._terminated(cur_node)
                return (cur_node == self.root_node or not if_terminated) and (not has_expanded) 
            
            elif isinstance(cur_node, (set, list)):
                to_remove = set()

                for idx, t_node in enumerate(cur_node):
                    if_terminated = self._terminated(t_node)
                    if not (t_node == self.root_node or not if_terminated):
                        to_remove.add(t_node)
                        ret_end_node_path[t_node] = rollout_path[idx]
                
                if len(to_remove) != 0: 
                    for t_node in to_remove:
                        cur_node.remove(t_node)
                        for i, path in enumerate(rollout_path):
                            if len(path) != 0 and path[-1] == t_node:
                                del rollout_path[i]

                    while len(rollout_path) > len(cur_node):
                        rollout_path.remove([])

                return (not has_expanded) and len(cur_node) != 0

        while terminate_fn(cur_node, rollout_path, has_expanded):
            if isinstance(cur_node, SeqNode): 
                self.nodes.add(cur_node)
            else: 
                self.nodes.update(cur_node)

            if isinstance(cur_node, SeqNode):
                if cur_node not in self.expanded_nodes:
                    assert (len(cur_node.children) == 0), "unexpanded node should have no child"
                    children = self._expand(cur_node)
                    self.all_nodes.update(children)
                    has_expanded = True 

                rollout_path = [rollout_path[0]+[nd] for nd in cur_node.children]
                cur_node = cur_node.children

            elif isinstance(cur_node, (set, list)):
                expand_node_set = set()

                for idx, t_node in enumerate(cur_node):
                    if t_node not in self.expanded_nodes:
                        expand_node_set.add(t_node)
                        assert (len(t_node.children) == 0), "unexpanded node should have no child"
                     
                if len(expand_node_set) > 0:
                    children = self._expand(expand_node_set)
                    for key, value in children.items():
                        self.all_nodes.update(value)
                    has_expanded = True

                _cur_node_list = list(cur_node)
                for idx, item in enumerate(_cur_node_list):
                    cur_node[idx] = max(item.children, key=lambda x:x.reward_score)
                    rollout_path[idx].append(cur_node[idx])

        for end_node in ret_end_node_path.keys():
            end_node.clear_kv()

            if end_node.parent is not None:
                end_node_parent = end_node.parent
                while end_node_parent:
                    if_terminated = [child_node.terminated for child_node in end_node_parent.children]
                    if all(if_terminated):
                        self.finished_node.add(end_node_parent)
                        end_node_parent.terminated = True
                        end_node_parent.clear_kv()
                        end_node_parent = end_node_parent.parent
                    else:
                        break 
       
        if isinstance(cur_node, SeqNode): 
            self.nodes.add(cur_node)
        
        if isinstance(cur_node, SeqNode):
            ret_end_node_path = {cur_node: rollout_path}
            return ret_end_node_path, cur_node, rollout_path
        elif isinstance(cur_node, (set, list)):
            return ret_end_node_path, ret_end_node_path.keys(), rollout_path
        
    def _reward(self, path, end_node):
        reward = self.reward_fn(end_node)
        return  0.2 if len(reward[0]) == 0 else reward[0][-1]
    
    def _backpropagate(self, end_node:SeqNode, reward:float):
        node = end_node
        while not node is None:
            node.Q += reward
            node.N += 1
            node = node.parent
        return 

    def _get_root_node(self, inputs) -> SeqNode:
        root_node = SeqNode(inputs)
        self.nodes.add(root_node)
        self.root_node = root_node
        return root_node
    
    def _expand(self, node):
        if isinstance(node, SeqNode): 
            self.nodes.add(node)
            # Track that this node has been expanded (needed for rollout logic)
            self.expanded_nodes.add(node)
            # Count every exploration (including re-explorations)
            self.total_node_explorations += 1
        else:
            for t_node in node: 
                self.nodes.add(t_node)
                # Track that this node has been expanded (needed for rollout logic)
                self.expanded_nodes.add(t_node)
                # Count every exploration (including re-explorations)
                self.total_node_explorations += 1
               
        children_nodes = self.expand_fn(node)
        if isinstance(node, SeqNode): 
            node.children = children_nodes
        else:
            for t_node, child_node in children_nodes.items():
                t_node.children = child_node

        return children_nodes

    def _best_node(self, topk=1, selected_paral_node={}):
        selective_node = list(filter(lambda i: i not in self.finished_node and i not in self.expanded_nodes and i not in selected_paral_node.keys(), self.all_nodes))

        if len(selective_node) <= topk:
            return set(selective_node)
        
        topk_node = heapq.nlargest(topk, selective_node, key=self._compute_ucb_score)
        return set(topk_node)
    
    def _compute_ucb_score(self, node:SeqNode):
        return node.Q / node.N + self.exploration_weight * math.sqrt(1+node.parent.N) * node.reward_score / (1+node.N)
    
    def _select_best_action(self, node:SeqNode):
        return self._compute_ucb_score(node)
    
    def _terminated(self, node):
        res = self.terminated_fn(node)
        if res:
            self.terminated_node.add(node)
            self.nodes.add(node)
            self.finished_node.add(node)
        return res

def _get_seq_length(self, layer_idx=0):
    return self.seen_seq_len

def get_past_kv(node, beam_size) -> DynamicCache:

    key, value = None, None

    cur_node = node
    while cur_node:
        past_key_values = cur_node.past_key_values if hasattr(cur_node, 'past_key_values') else None
        if past_key_values is not None and len(past_key_values.key_cache)  > 0: 
            if key is None: 
                key, value = [[] for _ in range(len(past_key_values))], [[] for _ in range(len(past_key_values))]
            for layer_idx in range(len(past_key_values)):
                key[layer_idx].append(past_key_values.key_cache[layer_idx])
                value[layer_idx].append(past_key_values.value_cache[layer_idx])
                
        cur_node = cur_node.parent

    if key is None:  return DynamicCache()
    
    key = [torch.cat(k[::-1] , dim=-2).repeat(beam_size, 1, 1, 1) for k in key]
    value = [torch.cat(v[::-1] , dim=-2).repeat(beam_size, 1, 1, 1) for v in value]

    new_past = DynamicCache()
    new_past.key_cache = key; new_past.value_cache = value
    
    return new_past


def find_shared_prefix_len(nodes):
    shared_predix_sets = [set() for _ in range(len(nodes))]
    for i, node in enumerate(nodes):
        cnt_node = node
        while cnt_node is not None:
            shared_predix_sets[i].add(cnt_node)
            cnt_node = cnt_node.parent

    intersection = set.intersection(*shared_predix_sets)
    kv_len = [node.get_all_kvlen() for node in intersection]
    return intersection, max(kv_len) 

def prepare_inputs_for_inference(node, tree_width, generation_config):
    past_key_values = DynamicCache()
    inputs = []
    kv_len = []

    for t_node in node:
        inputs.append(t_node.seq)
        t_past = get_past_kv(t_node, tree_width)
        
        if t_past is not None:
            past_key_values.key_cache.append(t_past.key_cache)
            past_key_values.value_cache.append(t_past.value_cache)
            kv_len.append(t_past.key_cache[0].shape[-2])
    
    input_len = [tensor.shape[-1] for tensor in inputs]
    max_input_len = max(input_len)
    max_kv_len = max(kv_len)

    padded_tensors = []
    masks = []
    # Use a default pad_token_id if None
    pad_token_id = generation_config.pad_token_id if generation_config.pad_token_id is not None else 0
    
    for tensor in inputs:
        # Ensure tensor is properly formatted
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device=tensor.device if hasattr(tensor, 'device') else 'cuda:0')
        
        # Ensure tensor has the right dtype for token IDs
        if tensor.dtype != torch.long:
            tensor = tensor.long()
        
        # Clamp token IDs to valid range to prevent CUDA errors
        vocab_size = 128256  # LLaMA vocab size
        tensor = torch.clamp(tensor, 0, vocab_size - 1)
        
        padding = torch.zeros([1, max_input_len - tensor.shape[-1]], dtype=tensor.dtype, device=tensor.device) 
        padded_tensors.append(torch.cat([padding + pad_token_id, tensor], dim=-1))
        # Create proper attention mask: 0 for padding, 1 for real tokens
        mask_padding = torch.zeros([1, max_input_len - tensor.shape[-1]], dtype=torch.bool, device=tensor.device)
        mask_tokens = torch.ones([1, tensor.shape[-1]], dtype=torch.bool, device=tensor.device)
        masks.append(torch.cat([mask_padding, mask_tokens], dim=-1))

    parallel_input = torch.cat(padded_tensors, dim=0)
    mask = torch.cat(masks, dim=0)
    
    # Ensure mask is properly formatted for attention
    if mask.dtype != torch.bool:
        mask = mask.bool()
    
    # Ensure mask has the right shape
    if mask.shape != parallel_input.shape:
        print(f"WARNING: Mask shape {mask.shape} doesn't match input shape {parallel_input.shape}")
        # Create a proper mask if shapes don't match
        mask = torch.ones_like(parallel_input, dtype=torch.bool)
    
    parallel_past_key_values = DynamicCache()
    for layer_idx in range(len(past_key_values.key_cache[0])):
        parallel_keys, parallel_values = [], []

        for parallel_idx in range(len(node)):
            key = past_key_values.key_cache[parallel_idx][layer_idx]
            value = past_key_values.value_cache[parallel_idx][layer_idx]

            padding_len = max_kv_len - key.shape[-2] 
            padding_shape = tuple(key.shape[:-2]) + (padding_len, ) + (key.shape[-1], )
            padding = torch.zeros(padding_shape, device=key.device, dtype=key.dtype)

            key = torch.cat((key, padding), dim=-2)
            value = torch.cat((value, padding), dim=-2)

            parallel_keys.append(key)
            parallel_values.append(value)

        parallel_keys = torch.cat(parallel_keys, dim=0)
        parallel_values = torch.cat(parallel_values, dim=0)
        parallel_past_key_values.update(parallel_keys, parallel_values, layer_idx)
    
    return parallel_input, parallel_past_key_values, mask, input_len, max_input_len, kv_len, max_kv_len 
        
def expand(node, generate_fn, reward_fn, inference_config, embedding_model=None, tokenizer=None, searcher=None) -> Set[SeqNode]:

    generation_config = inference_config.llm_config
    tree_width = inference_config.config.tree_width 

    if isinstance(node, SeqNode):
        node_seq = node.seq
        input_len = node_seq.shape[-1]
        past_key_values = get_past_kv(node, tree_width)

        output_dict = generate_fn(node_seq, past_key_values=past_key_values, cache_position=torch.tensor([input_len-1], \
                                                                             dtype=torch.int64, device=node.seq.device))

        past_key_values = output_dict.past_key_values
        assert isinstance(past_key_values, DynamicCache)
        output_ids = output_dict.sequences

        scores = torch.stack(output_dict.scores, dim=1)
        new_nodes = set()
        
        for i, (ids, sc) in enumerate(zip(output_ids.unbind(0), scores.unbind(0))):  
            not_pad_token = (ids != generation_config.pad_token_id)
            out = ids[not_pad_token]
            # Ensure out is a proper tensor
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, device=ids.device)
            elif out.dim() == 0:
                out = out.unsqueeze(0)
            
            # Create the sequence tensor properly
            seq_tensor = out.unsqueeze(0) if out.dim() == 1 else out
            n = SeqNode(seq=seq_tensor, len=out.shape[-1]-input_len, parent=node)

            # Fix: Handle both tensor and scalar cases
            if hasattr(not_pad_token, 'sum'):
                kv_len = not_pad_token.sum().item() - 1
            else:
                kv_len = 1 if not_pad_token else 0
            n.update_kv(past_key_values, n_seq_len=n.len, subtree_idx=i, kv_len=kv_len) 

            reward_score = reward_fn(n)
            n.reward_score = reward_score[0][-1] if len(reward_score[0]) > 0 else 0.2
            print(f"[SCORING DEBUG] Node {n.id} scored: {n.reward_score:.4f}")
            new_nodes.add(n)

        # Track clustering metrics if searcher is available
        nodes_before_clustering = len(new_nodes)
        
        # Apply clustering if embedding model is provided and clustering is enabled
        if (embedding_model is not None and 
            hasattr(inference_config.config, 'enable_clustering') and 
            inference_config.config.enable_clustering and 
            len(new_nodes) > 1):
            
            print(f"[EXPANSION DEBUG] Applying clustering to {nodes_before_clustering} nodes")
            
            # Get clustering parameters
            threshold = getattr(inference_config.config, 'clustering_threshold', 0.8)
            method = getattr(inference_config.config, 'clustering_method', 'cosine_similarity')
            enable_similarity_vote = getattr(inference_config.config, 'enable_similarity_vote', False)
            similarity_vote_weight = getattr(inference_config.config, 'similarity_vote_weight', 0.0)
            
            print(f"[EXPANSION DEBUG] Clustering config: threshold={threshold}, method={method}")
            print(f"[EXPANSION DEBUG] Similarity voting: enabled={enable_similarity_vote}, weight={similarity_vote_weight}")
            
            # Show nodes before clustering
            print(f"[EXPANSION DEBUG] Nodes before clustering: {[(n.id, getattr(n, 'reward_score', 'N/A')) for n in new_nodes]}")
            
            # Apply clustering
            new_nodes = cluster_parallel_nodes(new_nodes, embedding_model, threshold, method, tokenizer,
                                             enable_similarity_vote, similarity_vote_weight)
            
            nodes_after_clustering = len(new_nodes)
            nodes_pruned = nodes_before_clustering - nodes_after_clustering
            
            # Track clustering-specific metrics
            if searcher is not None:
                searcher.total_nodes_pruned += nodes_pruned
                searcher.clustering_applied_count += 1
                print(f"[CLUSTERING DEBUG] Applied clustering: pruned {nodes_pruned} nodes, total pruned: {searcher.total_nodes_pruned}, total applied: {searcher.clustering_applied_count}")
            
            print(f"[EXPANSION DEBUG] Nodes after clustering: {[(n.id, getattr(n, 'reward_score', 'N/A')) for n in new_nodes]}")
            print(f"[CLUSTERING METRICS] Pruned {nodes_pruned} nodes ({nodes_before_clustering} → {nodes_after_clustering})")
        else:
            # No clustering applied - all nodes are kept
            if searcher is not None:
                searcher.total_nodes_pruned += 0  # No nodes pruned when clustering is disabled
                print(f"[CLUSTERING DEBUG] No clustering applied: total pruned: {searcher.total_nodes_pruned}, total applied: {searcher.clustering_applied_count}")
            
            # Check if clustering was supposed to be enabled but couldn't be applied
            if (embedding_model is not None and 
                hasattr(inference_config.config, 'enable_clustering') and 
                inference_config.config.enable_clustering):
                print(f"[EXPANSION DEBUG] Clustering enabled but only {len(new_nodes)} node(s), skipping clustering")
            else:
                print(f"[EXPANSION DEBUG] Clustering disabled or no embedding model, using {len(new_nodes)} nodes as-is")

        return new_nodes

    else:
        
        ordered_nodes = list(node)
        parallel_input, parallel_past_key_values, mask, input_len, max_input_len, kv_len, max_kv_len = \
            prepare_inputs_for_inference(ordered_nodes, tree_width, generation_config)

        if hasattr(parallel_past_key_values, "get_seq_length") and parallel_past_key_values.get_seq_length() is not None:
            parallel_past_key_values.seen_seq_len = max_input_len - 1
            parallel_past_key_values.get_seq_length = MethodType(_get_seq_length, parallel_past_key_values)
            
        output_dict = generate_fn(parallel_input, attention_mask=mask, past_key_values=parallel_past_key_values)
      
        past_key_values = output_dict.past_key_values
        assert isinstance(past_key_values, DynamicCache)
        output_ids = output_dict.sequences
        scores = torch.stack(output_dict.scores, dim=1)

        grouped_child_nodes = {t_node: set() for t_node in ordered_nodes}
        
        if len(output_ids.unbind(0)) < 1: import pdb; pdb.set_trace()

        for i, (ids, sc) in enumerate(zip(output_ids.unbind(0), scores.unbind(0))): 
            not_pad_token = (ids != generation_config.pad_token_id)
            out = ids[not_pad_token]
            
            # Ensure out is a proper tensor before creating SeqNode
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, device=ids.device)
            elif out.dim() == 0:
                out = out.unsqueeze(0)
            
            # Create the sequence tensor properly
            seq_tensor = out.unsqueeze(0) if out.dim() == 1 else out
            n = SeqNode(seq=seq_tensor, len=out.shape[-1] - input_len[i//tree_width], parent=ordered_nodes[i//tree_width])

            # Fix: Handle both tensor and scalar cases
            if hasattr(not_pad_token, 'sum'):
                kv_len = not_pad_token.sum().item() - 1
            else:
                kv_len = 1 if not_pad_token else 0
            n.update_kv(past_key_values, n_seq_len=n.len, subtree_idx=i, kv_len=kv_len)

            reward_score = reward_fn(n)
            n.reward_score = reward_score[0][-1] if len(reward_score[0]) > 0 else 0.2
            grouped_child_nodes[ordered_nodes[i//tree_width]].add(n)

        return grouped_child_nodes


def reward(node: SeqNode, model, reward_model, inference_config):
    if reward_model is None:
        return [[0.2]]
    else:
        # Decode tensor to list of strings
        # Work with a copy to avoid modifying the original node.seq
        seq_to_decode = node.seq
        
        # Handle different input types more robustly
        if isinstance(seq_to_decode, list):
            # If it's a list, convert to tensor
            if len(seq_to_decode) > 0 and isinstance(seq_to_decode[0], list):
                # Handle nested lists (list of lists)
                seq_to_decode = torch.tensor(seq_to_decode, device=next(model.parameters()).device)
            else:
                # Handle simple list
                seq_to_decode = torch.tensor([seq_to_decode], device=next(model.parameters()).device)
        elif not isinstance(seq_to_decode, torch.Tensor):
            # Convert other types to tensor
            seq_to_decode = torch.tensor([seq_to_decode], device=next(model.parameters()).device)
        else:
            # If it's already a tensor, make a copy to avoid modifying the original
            seq_to_decode = seq_to_decode.clone()
        
        # Ensure proper shape for batch_decode (should be 2D: [batch_size, seq_length])
        if seq_to_decode.dim() == 1:
            seq_to_decode = seq_to_decode.unsqueeze(0)
        elif seq_to_decode.dim() == 0:
            seq_to_decode = seq_to_decode.unsqueeze(0).unsqueeze(0)
        
        # Ensure it's the right dtype (integer tokens)
        if seq_to_decode.dtype != torch.long:
            seq_to_decode = seq_to_decode.long()
        
        inputs = model.tokenizer.batch_decode(seq_to_decode, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return reward_model.score(inputs, inference_config)

def terminated(node: SeqNode, inputs, inference_config) -> bool:
    generation_config = inference_config.llm_config
    end_token_id = node.seq[0,-1].item()
    new_tokens = node.seq.shape[-1] - inputs.shape[-1]

    node.terminated = end_token_id in generation_config.eos_token_id or new_tokens >= inference_config.config.max_new_tokens
    return end_token_id in generation_config.eos_token_id or new_tokens >= inference_config.config.max_new_tokens

global model

def dpts_generate_cot(self, inputs, inference_config, embedding_model=None):
    generation_config = inference_config.llm_config
    config = inference_config.config
    global model
    model = self
  
    generate_fn = partial(
        self.generate_step, 
        inference_config=inference_config, 
        num_return_sequences=config.tree_width, 
        return_dict_in_generate=True, 
        output_scores=True, 
        use_cache=True
        )
    reward_fn = partial(reward, model=self.generation_model, reward_model=self.reward_model, inference_config=inference_config)
    terminated_fn = partial(terminated, inputs=inputs, inference_config=inference_config)
    # Get tokenizer for clustering
    tokenizer = getattr(self.generation_model, 'tokenizer', None)
    # Create a wrapper function that can access the searcher
    def expand_with_searcher(node):
        return expand(node, generate_fn=generate_fn, reward_fn=reward_fn, inference_config=inference_config, 
                     embedding_model=embedding_model, tokenizer=tokenizer, searcher=expand_with_searcher.searcher)
    
    importance_fn = partial(self.importance, inference_config=inference_config)
    
    dpts_searcher = CoT_DPTS_Searcher(config.max_rollout, config.max_step_time, expand_with_searcher, reward_fn, terminated_fn, \
                                        importance_fn=importance_fn, inference_config=inference_config, generation_config=generation_config)
    
    # Set the searcher reference after creation
    expand_with_searcher.searcher = dpts_searcher

    try:
        paths, scores = dpts_searcher.process(inputs)
    except torch.cuda.OutOfMemoryError:
        paths, scores = dpts_searcher.finalize()

    # Get clustering metrics before deleting the searcher and store in model
    clustering_metrics = dpts_searcher.get_clustering_metrics()
    self._last_clustering_metrics = clustering_metrics
    
    del dpts_searcher
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return paths, scores
