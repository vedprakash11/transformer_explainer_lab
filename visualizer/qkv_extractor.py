"""
Utility functions to extract Query, Key, and Value vectors from transformer models.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from transformers import BertModel, AutoModelForCausalLM


def extract_qkv_bert(
    model: BertModel,
    inputs: dict,
    layer: int,
    head: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Q, K, V vectors from a BERT model for a specific layer and head.
    
    Args:
        model: BERT model instance
        inputs: Tokenized inputs dictionary
        layer: Layer index
        head: Head index
    
    Returns:
        Tuple of (Q, K, V) tensors, each of shape (seq_len, head_dim)
    """
    with torch.no_grad():
        # Get embeddings
        embedding_output = model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids", None),
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        
        # Forward through layers up to target layer
        hidden_states = embedding_output
        for i in range(layer):
            layer_outputs = model.encoder.layer[i](
                hidden_states,
                attention_mask=inputs.get("attention_mask", None),
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        
        # Get encoder layer and attention module
        encoder_layer = model.encoder.layer[layer]
        attention = encoder_layer.attention.self
        
        # Compute Q, K, V
        query = attention.query(hidden_states)
        key = attention.key(hidden_states)
        value = attention.value(hidden_states)
        
        # Reshape for multi-head attention
        batch_size, seq_len, hidden_size = query.shape
        num_heads = attention.num_attention_heads
        head_dim = hidden_size // num_heads
        
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Select specific head and remove batch dimension
        q = query[0, head, :, :]  # (seq_len, head_dim)
        k = key[0, head, :, :]    # (seq_len, head_dim)
        v = value[0, head, :, :]   # (seq_len, head_dim)
        
        return q, k, v


def extract_qkv_llama(
    model: AutoModelForCausalLM,
    inputs: dict,
    layer: int,
    head: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Q, K, V vectors from a Llama model for a specific layer and head.
    
    Args:
        model: Llama model instance
        inputs: Tokenized inputs dictionary
        layer: Layer index
        head: Head index
    
    Returns:
        Tuple of (Q, K, V) tensors, each of shape (seq_len, head_dim)
    """
    with torch.no_grad():
        # Get model base (transformer)
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
        
        # Get embeddings
        if hasattr(base_model, 'embed_tokens'):
            hidden_states = base_model.embed_tokens(inputs["input_ids"])
        else:
            # Fallback for different architectures
            hidden_states = base_model.get_input_embeddings()(inputs["input_ids"])
        
        # Process through layers up to target layer
        for i in range(layer):
            if hasattr(base_model, 'layers'):
                hidden_states = base_model.layers[i](hidden_states)[0]
            elif hasattr(base_model, 'h'):
                hidden_states = base_model.h[i](hidden_states)[0]
        
        # Get target layer
        if hasattr(base_model, 'layers'):
            target_layer = base_model.layers[layer]
        elif hasattr(base_model, 'h'):
            target_layer = base_model.h[layer]
        else:
            raise ValueError("Could not find layer structure in model")
        
        # Get self-attention
        if hasattr(target_layer, 'self_attn'):
            self_attn = target_layer.self_attn
        elif hasattr(target_layer, 'attention'):
            self_attn = target_layer.attention
        else:
            raise ValueError("Could not find attention module in layer")
        
        # Extract Q, K, V
        if hasattr(self_attn, 'q_proj'):
            query = self_attn.q_proj(hidden_states)
            key = self_attn.k_proj(hidden_states)
            value = self_attn.v_proj(hidden_states)
        elif hasattr(self_attn, 'query'):
            query = self_attn.query(hidden_states)
            key = self_attn.key(hidden_states)
            value = self_attn.value(hidden_states)
        else:
            raise ValueError("Could not find Q, K, V projections")
        
        # Reshape for multi-head attention
        batch_size, seq_len, hidden_size = query.shape
        
        # Try to get num_heads from config or infer
        if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        elif hasattr(model, 'config') and hasattr(model.config, 'num_heads'):
            num_heads = model.config.num_heads
        else:
            # Infer from hidden_size (common head_dim is 64 or 128)
            num_heads = hidden_size // 64
        
        head_dim = hidden_size // num_heads
        
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Select specific head and remove batch dimension
        q = query[0, head, :, :]  # (seq_len, head_dim)
        k = key[0, head, :, :]    # (seq_len, head_dim)
        v = value[0, head, :, :]   # (seq_len, head_dim)
        
        return q, k, v


def extract_qkv(
    model,
    inputs: dict,
    layer: int,
    head: int,
    model_type: str = "bert"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Q, K, V vectors from a transformer model.
    
    Args:
        model: Model instance
        inputs: Tokenized inputs dictionary
        layer: Layer index
        head: Head index
        model_type: Type of model ("bert" or "llama")
    
    Returns:
        Tuple of (Q, K, V) tensors, each of shape (seq_len, head_dim)
    """
    try:
        if model_type == "bert":
            return extract_qkv_bert(model, inputs, layer, head)
        elif model_type == "llama":
            return extract_qkv_llama(model, inputs, layer, head)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"Error extracting Q, K, V vectors: {str(e)}") from e

