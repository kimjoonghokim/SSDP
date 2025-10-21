from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch
from transformers.generation import GenerationConfig


class BaseEmbeddingModel(ABC):
    """Base class for embedding models used in node clustering."""
    
    def __init__(self, model_path: str, device: torch.device, **kwargs):
        """
        Initialize the embedding model.
        
        Args:
            model_path: Path to the embedding model
            device: Device to run the model on
            **kwargs: Additional model-specific parameters
        """
        self.model_path = model_path
        self.device = device
        self.config = self._default_config()
        
    def _default_config(self) -> GenerationConfig:
        """Return default configuration for the embedding model."""
        config = GenerationConfig()
        # Add embedding-specific config parameters
        config.enable_clustering = False
        config.clustering_threshold = None
        config.clustering_method = None
        config.embedding_model = None
        config.embedding_model_path = None
        return config
    
    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing embeddings
        """
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> torch.Tensor:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Tensor of shape (embedding_dim,) containing the embedding
        """
        pass
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize embeddings
        embedding1 = embedding1 / torch.norm(embedding1, dim=-1, keepdim=True)
        embedding2 = embedding2 / torch.norm(embedding2, dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.dot(embedding1, embedding2).item()
        return max(0.0, similarity)  # Ensure non-negative
    
    def batch_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarities for a batch of embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Tensor of shape (batch_size, batch_size) with pairwise similarities
        """
        # Normalize embeddings
        normalized = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        # Ensure non-negative values
        similarity_matrix = torch.clamp(similarity_matrix, min=0.0)
        
        return similarity_matrix
