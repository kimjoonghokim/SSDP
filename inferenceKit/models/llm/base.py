from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prm import BaseProcessRewardModel
from inferenceKit.utils import update_single_config, update_sampling_params_from_generation_config

# Conditional vllm import
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

class BaseLargeLanguageModel(nn.Module):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        self.device = kwargs.pop("device", torch.cuda.current_device)
        self.device_map = kwargs.pop("device_map", None) or self.device
        
        self.base_config = self._load_config(model_path)
        self.config = self._default_config(self.base_config)
        self.config, _, _ = update_single_config(self.config, copy=True, **kwargs)
        
        self.model_path = model_path
        self.model, self.tokenizer = self._load_model(model_path)
    
    def _load_config(self, model_path):
        try:
            config = GenerationConfig.from_pretrained(model_path)
        except:
            config = GenerationConfig()
        return config
    
    def _load_model(self, model_path):
        if model_path != None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=self.device_map, 
                torch_dtype=self.config.dtype, 
                attn_implementation="flash_attention_2" if self.config.flash_attn else None,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Ensure tokenizer has proper pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        return model, tokenizer

    def _default_config(self, base_config=None):
        config = base_config or GenerationConfig()
        
        # Ensure proper configuration for LLaMA models
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = 0  # Default pad token for LLaMA
        
        if not hasattr(config, 'eos_token_id') or config.eos_token_id is None:
            config.eos_token_id = 2  # Default EOS token for LLaMA
            
        return config

    @torch.inference_mode()
    def generate(self, inputs, generation_config: GenerationConfig, **kwargs):
        outputs = self.model.generate(inputs, generation_config, **kwargs)
        return outputs

class DefaultLargeLanguageModel(BaseLargeLanguageModel):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        
    
    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        
        config.do_sample = True
        config.num_beams = 4
        config.max_new_tokens = 2048
        return config

    @torch.inference_mode()
    def generate(self, inputs, generation_config: GenerationConfig, **kwargs):
        return self.model.generate(inputs, generation_config, tokenizer=self.tokenizer, **kwargs)
      

class VLLMLargeLanguageModel(BaseLargeLanguageModel):
    def __init__(self, model_path, **kwargs):
        if not VLLM_AVAILABLE:
            raise ImportError("vllm is not available. Please install vllm or use a different model type.")
        super().__init__(model_path, **kwargs)
        
    def _load_model(self, model_path):
        if model_path != None:
            model = LLM(
                model_path, 
                dtype=self.config.dtype, 
                # skip_tokenizer_init=True
                )
            tokenizer = model.get_tokenizer()
        return model, tokenizer

    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        
        config.do_sample = True
        config.num_beams = 4
        config.max_new_tokens = 2048
        return config

    @torch.inference_mode()
    def generate(self, inputs, generation_config: GenerationConfig, **kwargs):
        samplingParams = SamplingParams(**kwargs)
        samplingParams = update_sampling_params_from_generation_config(samplingParams, generation_config)
        return self.model.generate(inputs, samplingParams, use_tqdm=False)