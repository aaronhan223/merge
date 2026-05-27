"""
Checkpoint utility functions for loading and creating models from checkpoints.
"""

import argparse
import torch
from typing import Dict, List, Tuple, Any
from model.trus_moe_multimodal import MultimodalTRUSMoEModel


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, Any], Any, List[Dict], List[str], float]:
    """
    Load model checkpoint and return model, args, and other metadata.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        
    Returns:
        Tuple of (model_state_dict, args, modality_configs, modality_names, best_metric)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Add argparse.Namespace to safe globals for PyTorch 2.6+ compatibility
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract checkpoint information
    epoch = checkpoint['epoch']
    model_state_dict = checkpoint['model_state_dict']
    args = checkpoint['args']
    modality_configs = checkpoint['modality_configs']
    modality_names = checkpoint['modality_names']
    
    # Handle different metric names (accuracy vs auc)
    if 'best_val_acc' in checkpoint:
        best_metric = checkpoint['best_val_acc']
        metric_name = "accuracy"
    elif 'best_val_auc' in checkpoint:
        best_metric = checkpoint['best_val_auc']
        metric_name = "AU-ROC"
    else:
        best_metric = 0.0
        metric_name = "metric"
    
    print(f"Checkpoint info:")
    print(f"  Best epoch: {epoch + 1}")
    print(f"  Best validation {metric_name}: {best_metric:.4f}")
    print(f"  Modalities: {modality_names}")
    
    return model_state_dict, args, modality_configs, modality_names, best_metric


def create_model_from_checkpoint(model_state_dict: Dict[str, Any], 
                                args: Any, 
                                modality_configs: List[Dict], 
                                num_classes: int, 
                                device: torch.device) -> MultimodalTRUSMoEModel:
    """
    Create model instance from checkpoint information.
    
    Args:
        model_state_dict: Model state dictionary from checkpoint
        args: Arguments from checkpoint
        modality_configs: Modality configurations
        num_classes: Number of output classes
        device: Device to create model on
        
    Returns:
        Loaded and configured model
    """
    # MoE configuration
    moe_router_config = {
        "gru_hidden_dim": args.moe_router_gru_hidden_dim,
        "token_processed_dim": args.moe_router_token_processed_dim,
        "attn_key_dim": args.moe_router_attn_key_dim,
        "attn_value_dim": args.moe_router_attn_value_dim,
    }
    
    moe_layer_config = {
        "num_experts": args.moe_num_experts,
        "num_synergy_experts": args.moe_num_synergy_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "synergy_expert_nhead": args.nhead,
        "router_config": moe_router_config,
        "capacity_factor": args.moe_capacity_factor,
        "drop_tokens": args.moe_drop_tokens,
    }

    # Create model
    model = MultimodalTRUSMoEModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_classes=num_classes,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_checkpoint=args.use_gradient_checkpointing,
        output_attention=False
    ).to(device)
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"Model created and loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model
