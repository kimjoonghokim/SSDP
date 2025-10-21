from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingModel


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Sentence-Transformers based embedding model for node clustering."""
    
    def __init__(self, model_path: str, device: torch.device, **kwargs):
        """
        Initialize Sentence-Transformers embedding model.
        
        Args:
            model_path: Path or name of the sentence-transformers model
            device: Device to run the model on
            **kwargs: Additional parameters (batch_size, max_seq_length, etc.)
        """
        super().__init__(model_path, device, **kwargs)
        
        # Default parameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_seq_length = kwargs.get('max_seq_length', 512)
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        
        # Load the model
        self.model = SentenceTransformer(model_path, device=str(device))
        
        # Set model parameters
        self.model.max_seq_length = self.max_seq_length
        
        # Move to device
        self.model = self.model.to(device)
        self.model.eval()
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing embeddings
        """
        if not texts:
            return torch.empty(0, 0, device=self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                device=self.device
            )
        
        return embeddings
    
    def encode_single(self, text: str) -> torch.Tensor:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Tensor of shape (embedding_dim,) containing the embedding
        """
        return self.encode([text])[0]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Get embedding dimension by encoding a dummy text
        dummy_embedding = self.encode_single("dummy")
        return dummy_embedding.shape[0]
