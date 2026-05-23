import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

def calculate_expert_activation_ratios_time_moe(
    expert_indices: torch.Tensor,
    num_experts: int,
    modality_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Calculate the activation ratio for each expert per modality for TIME-MoE model.
    
    Args:
        expert_indices: Tensor of shape (B, M, T, k) containing top-k expert indices
        num_experts: Total number of experts
        modality_names: Optional list of modality names
        
    Returns:
        activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
    """
    B, M, T, k = expert_indices.shape
    
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(M)]
    
    # Initialize activation counts
    activation_counts = np.zeros((num_experts, M))
    
    # Count activations for each expert and modality
    for b in range(B):
        for m in range(M):
            for t in range(T):
                for j in range(k):
                    expert_idx = expert_indices[b, m, t, j].item()
                    activation_counts[expert_idx, m] += 1
    
    # Calculate total tokens per modality
    total_tokens_per_modality = B * T * k
    
    # Calculate activation ratios (percentages)
    activation_ratios = (activation_counts / total_tokens_per_modality) * 100
    
    return activation_ratios


# def plot_expert_activation_histogram(
#     activation_ratios: np.ndarray,
#     modality_names: List[str],
#     layer_name: str,
#     model_type: str,
#     save_path: Optional[str] = None,
#     figsize: Tuple[int, int] = (12, 8)
# ):
#     """
#     Plot a histogram showing activation ratios for each expert.
    
#     Args:
#         activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
#         modality_names: List of modality names
#         layer_name: Name of the MoE layer
#         model_type: "TIME-MoE" or "Baseline"
#         save_path: Optional path to save the figure
#         figsize: Figure size
#     """
#     num_experts, num_modalities = activation_ratios.shape
    
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Set up bar positions
#     x = np.arange(num_experts)
#     width = 0.8 / num_modalities
    
#     # Create color palette
#     colors = sns.color_palette("husl", num_modalities)
    
#     # Plot bars for each modality
#     for i, modality in enumerate(modality_names):
#         offset = (i - num_modalities / 2) * width + width / 2
#         bars = ax.bar(x + offset, activation_ratios[:, i], width, 
#                       label=modality, color=colors[i], alpha=0.8)
        
#         # Add value labels on bars if they're significant
#         # Original code:
#         for bar in bars:
#             height = bar.get_height()
#             if height > 1:  # Only show labels for bars > 1%
#                 ax.text(bar.get_x() + bar.get_width()/2., height,
#                        f'{height:.1f}', ha='center', va='bottom', fontsize=22) # fontsize=8)
        
#         # New code with improved font sizing and overlap prevention (commented out):
#         # # Store label positions to avoid overlaps
#         # label_positions = []
#         # 
#         # for j, bar in enumerate(bars):
#         #     height = bar.get_height()
#         #     if height > 1:  # Only show labels for bars > 1%
#         #         # Calculate available space for the label
#         #         bar_width = bar.get_width()
#         #         bar_center_x = bar.get_x() + bar_width/2.
#         #         
#         #         # Use larger font size but adjust positioning to avoid overlaps
#         #         fontsize = 12  # Reduced from 14 to help with spacing
#         #         
#         #         # For bars that are too narrow or too short, use smaller font
#         #         if bar_width < 0.15:  # Very narrow bars
#         #             fontsize = 10
#         #         elif height < 3:  # Very short bars
#         #             fontsize = 10
#         #         
#         #         # Calculate initial text position
#         #         base_text_y = height + 1.0  # Increased offset
#         #         text_y = base_text_y
#         #         
#         #         # Check for overlaps with existing labels and adjust vertically
#         #         min_separation = 3.0  # Minimum vertical separation between labels
#         #         for prev_x, prev_y in label_positions:
#         #             # If labels are horizontally close, stack them vertically
#         #             if abs(bar_center_x - prev_x) < bar_width * 2:  # Horizontally close
#         #                 if abs(text_y - prev_y) < min_separation:
#         #                     text_y = max(text_y, prev_y + min_separation)
#         #         
#         #         # Add rotation for very crowded areas or when stacked high
#         #         rotation = 0
#         #         if bar_width < 0.1 or text_y > height + 8:  # Very crowded or stacked high
#         #             rotation = 45
#         #             fontsize = min(fontsize, 10)  # Use smaller font when rotated
#         #         
#         #         # Store this label's position
#         #         label_positions.append((bar_center_x, text_y))
#         #         
#         #         ax.text(bar_center_x, text_y,
#         #                f'{height:.1f}', ha='center', va='bottom', 
#         #                fontsize=fontsize, rotation=rotation, 
#         #                fontweight='bold', color='black')
    
#     # Customize the plot
#     ax.set_xlabel('Expert Index', fontsize=25)
#     ax.set_ylabel('Activation Ratio (%)', fontsize=25)
#     ax.set_title(f'{model_type} Model - {layer_name} Expert Activation Ratios', fontsize=25)
#     ax.set_xticks(x)
#     ax.set_xticklabels([f'{i}' for i in range(num_experts)])
#     ax.legend(title='Modality', loc='upper right', fontsize=20, title_fontsize=22) # bbox_to_anchor=(1.05, 1), loc='upper left')
#     # ax.grid(True, alpha=0.3, axis='y')
    
#     # Set y-axis limit with extra space for text labels
#     max_value = activation_ratios.max()
#     ax.set_ylim(0, max(100, max_value * 1.1))  # Extra 10% space for labels
#     # Set font sizes for ticks
#     ax.tick_params(axis='x', labelsize=20)
#     ax.tick_params(axis='y', labelsize=20)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to {save_path}")
    
#     plt.show()

from adjustText import adjust_text

def plot_expert_activation_histogram(
    activation_ratios: np.ndarray,
    modality_names: List[str],
    layer_name: str,
    model_type: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    moe_num_synergy_experts: Optional[int] = None
):
    num_experts, num_modalities = activation_ratios.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(num_experts)
    width = 0.8 / num_modalities
    
    colors = sns.color_palette("husl", num_modalities)
    
    all_texts = []  # collect all labels for adjustText
    
    for i, modality in enumerate(modality_names):
        offset = (i - num_modalities / 2) * width + width / 2
        bars = ax.bar(x + offset, activation_ratios[:, i], width, 
                      label=modality, color=colors[i], alpha=0.8)
        
        # Collect text labels for adjustText
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 1:  # Only show labels for bars > 1%
                text = ax.text(
                    bar.get_x() + bar.get_width()/2.,  # x pos
                    height + 1.0,                      # y pos with small offset
                    f'{height:.1f}', ha='center', va='bottom',
                    fontsize=20, color='black'
                )
                all_texts.append(text)
    
     # Adjust text positions to avoid overlaps
    adjust_text(all_texts,
                ax=ax,
                only_move={'points':'y', 'texts':'y'},
                force_points=0.05,
                force_text=0.05,
                expand_points=(1.0, 1.4),
                expand_text=(1.0, 1.4)
                # arrowprops=dict(arrowstyle='-', color='black', lw=0.5)
                )
     
    # Highlight synergy experts if specified
    if moe_num_synergy_experts is not None and moe_num_synergy_experts > 0:
        # Get current x-tick labels and modify synergy expert labels
        current_labels = [f'{i}' for i in range(num_experts)]
        for expert_idx in range(min(moe_num_synergy_experts, num_experts)):
            current_labels[expert_idx] = f'{expert_idx}'
        
        # Set the modified labels with color highlighting
        ax.set_xticklabels(current_labels)
        
        # Color the synergy expert tick labels red
        for expert_idx in range(min(moe_num_synergy_experts, num_experts)):
            ax.get_xticklabels()[expert_idx].set_color('red')
            ax.get_xticklabels()[expert_idx].set_fontweight('bold')

    # Customize the plot
    ax.set_xlabel('Expert Index', fontsize=25)
    ax.set_ylabel('Activation Ratio (%)', fontsize=25)
    ax.set_title(f'{model_type} Model - {layer_name} Expert Activation Ratios', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_experts)])
    ax.legend(title='Modality', loc='upper right', fontsize=20, title_fontsize=22)
    
    max_value = activation_ratios.max()
    ax.set_ylim(0, max(100, max_value * 1.4))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()



def plot_all_moe_layers_time_moe(
    model: nn.Module,
    data_batch,
    rus_values: Dict[str, torch.Tensor],
    modality_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    moe_num_synergy_experts: Optional[int] = None
):
    """
    Plot activation histograms for all MoE layers in a TIME-MoE model.
    
    Args:
        model: The TIME-MoE MoE model
        data_batch: Input data batch - can be either:
                   - Single tensor of shape (B, M, T, E)
                   - List of tensors [tensor1, tensor2, ...] where each tensor is (B, T, E_i)
        rus_values: RUS values dictionary
        modality_names: Optional list of modality names
        save_dir: Optional directory to save figures
    """
    model.eval()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Forward pass to get all auxiliary outputs
        _, all_aux_moe_outputs = model(data_batch, rus_values)
    
    # Get modality names and count from data_batch
    if isinstance(data_batch, list):
        num_modalities = len(data_batch)
    else:
        num_modalities = data_batch.shape[1]
    
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(num_modalities)]
    
    # Plot for each MoE layer
    for layer_idx, aux_outputs in enumerate(all_aux_moe_outputs):
        expert_indices = aux_outputs['expert_indices']
        
        # Get the actual MoE layer to find number of experts
        moe_layer_count = 0
        num_experts = 8  # Default fallback
        
        # Handle DDP wrapped models
        actual_model = model.module if hasattr(model, 'module') else model
        
        for layer in actual_model.layers:
            if hasattr(layer, 'moe_layer'):
                if moe_layer_count == layer_idx:
                    num_experts = layer.moe_layer.num_experts
                    break
                moe_layer_count += 1
        
        # Calculate activation ratios
        activation_ratios = calculate_expert_activation_ratios_time_moe(
            expert_indices, num_experts, modality_names
        )
        
        # Plot histogram
        layer_name = f"MoE Layer {layer_idx + 1}"
        save_path = os.path.join(save_dir, f"time_moe_layer_{layer_idx + 1}.pdf") if save_dir else None
        
        plot_expert_activation_histogram(
            activation_ratios,
            modality_names,
            layer_name,
            "TIME-MoE",
            save_path,
            moe_num_synergy_experts=moe_num_synergy_experts
        )
        
        # Plot stacked activation plot
        stacked_save_path = os.path.join(save_dir, f"time_moe_layer_{layer_idx + 1}_stacked.pdf") if save_dir else None
        
        create_stacked_activation_plot(
            activation_ratios,
            modality_names,
            layer_name,
            "TIME-MoE",
            stacked_save_path,
            moe_num_synergy_experts=moe_num_synergy_experts
        )


def plot_all_moe_layers_baseline(
    model: nn.Module,
    data_batch,
    modality_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    moe_num_synergy_experts: Optional[int] = None,
    seed: int = None,
    subject: int = None
):
    """
    Plot activation histograms for all MoE layers in a baseline model.
    
    Args:
        model: The baseline MoE model
        data_batch: Input data batch - can be either:
                   - Single tensor of shape (B, M, T, E)
                   - List of tensors [tensor1, tensor2, ...] where each tensor is (B, T, E_i)
        modality_names: Optional list of modality names
        save_dir: Optional directory to save figures
    """
    model.eval()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Handle both tensor and list input formats
    if isinstance(data_batch, list):
        B, T = data_batch[0].shape[:2]
        M = len(data_batch)
    else:
        B, M, T, E = data_batch.shape
    
    with torch.no_grad():
        # Forward pass to get all auxiliary outputs
        _, all_aux_moe_outputs = model(data_batch)
    
    # Get modality names if not provided
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(M)]
    
    # Plot for each MoE layer
    for layer_idx, aux_outputs in enumerate(all_aux_moe_outputs):
        expert_indices = aux_outputs['expert_indices']
        
        # Get the actual MoE layer to find number of experts
        moe_layer_count = 0
        num_experts = 8  # Default fallback
        
        # Handle DDP wrapped models
        actual_model = model.module if hasattr(model, 'module') else model
        
        for layer in actual_model.layers:
            if hasattr(layer, 'moe_layer'):
                if moe_layer_count == layer_idx:
                    num_experts = layer.moe_layer.num_experts
                    break
                moe_layer_count += 1
        
        # Calculate activation ratios
        activation_ratios = calculate_expert_activation_ratios_time_moe(
            expert_indices, num_experts, modality_names
        )
        
        # Plot histogram
        layer_name = f"MoE Layer {layer_idx + 1}"
        save_path = os.path.join(save_dir, f"baseline_moe_layer_{layer_idx + 1}_seed{seed}_sub{subject}.pdf") if save_dir else None
        
        plot_expert_activation_histogram(
            activation_ratios,
            modality_names,
            layer_name,
            "Baseline",
            save_path,
            moe_num_synergy_experts=moe_num_synergy_experts
        )
        
        # Plot stacked activation plot
        stacked_save_path = os.path.join(save_dir, f"baseline_moe_layer_{layer_idx + 1}_seed{seed}_sub{subject}_stacked.pdf") if save_dir else None
        
        create_stacked_activation_plot(
            activation_ratios,
            modality_names,
            layer_name,
            "Baseline",
            stacked_save_path,
            moe_num_synergy_experts=moe_num_synergy_experts
        )


def create_stacked_activation_plot(
    activation_ratios: np.ndarray,
    modality_names: List[str],
    layer_name: str,
    model_type: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    moe_num_synergy_experts: Optional[int] = None
):
    """
    Create a stacked bar plot showing the composition of each expert's activations.
    
    Args:
        activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
        modality_names: List of modality names
        layer_name: Name of the MoE layer
        model_type: "TIME-MoE" or "Baseline"
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    num_experts, num_modalities = activation_ratios.shape
    
    # Normalize to show composition (percentage of each expert's total activation)
    expert_totals = activation_ratios.sum(axis=1, keepdims=True)
    expert_totals[expert_totals == 0] = 1  # Avoid division by zero
    normalized_ratios = (activation_ratios / expert_totals) * 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color palette
    colors = sns.color_palette("husl", num_modalities)
    
    # Create stacked bars
    x = np.arange(num_experts)
    bottom = np.zeros(num_experts)
    
    for i, modality in enumerate(modality_names):
        ax.bar(x, normalized_ratios[:, i], bottom=bottom, 
               label=modality, color=colors[i], alpha=0.8)
        bottom += normalized_ratios[:, i]
    
    # Highlight synergy experts if specified
    if moe_num_synergy_experts is not None and moe_num_synergy_experts > 0:
        # Get current x-tick labels and modify synergy expert labels
        current_labels = [f'{i}' for i in range(num_experts)]
        for expert_idx in range(min(moe_num_synergy_experts, num_experts)):
            current_labels[expert_idx] = f'{expert_idx}'
        
        # Set the modified labels with color highlighting
        ax.set_xticklabels(current_labels)
        
        # Color the synergy expert tick labels red
        for expert_idx in range(min(moe_num_synergy_experts, num_experts)):
            ax.get_xticklabels()[expert_idx].set_color('red')
            ax.get_xticklabels()[expert_idx].set_fontweight('bold')
    
    # Customize the plot
    ax.set_xlabel('Expert Index', fontsize=25)
    ax.set_ylabel('Modality Composition (%)', fontsize=25)
    ax.set_title(f'{model_type} Model - {layer_name} Expert Modality Composition', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_experts)])
    ax.legend(title='Modality', loc='upper right', fontsize=24, title_fontsize=25) # bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    # ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Example usage function
def analyze_expert_activations(
    time_moe_model: Optional[nn.Module] = None,
    baseline_model: Optional[nn.Module] = None,
    data_batch = None,
    rus_values: Optional[Dict[str, torch.Tensor]] = None,
    modality_names: Optional[List[str]] = None,
    save_dir: str = "../results/expert_activation_plots",
    moe_num_synergy_experts: Optional[int] = None,
    seed: int = None,
    subject: int = None
):
    """
    Analyze and plot expert activations for both TIME-MoE and baseline models.
    
    Args:
        time_moe_model: TIME-MoE model (optional)
        baseline_model: Baseline MoE model (optional)
        data_batch: Input data batch - can be either:
                   - Single tensor of shape (B, M, T, E)
                   - List of tensors [tensor1, tensor2, ...] where each tensor is (B, T, E_i)
        rus_values: RUS values dictionary (required for TIME-MoE model)
        modality_names: Optional list of modality names
        save_dir: Directory to save plots
    """
    if data_batch is None:
        print("Error: data_batch is required")
        return
    
    if time_moe_model is not None:
        if rus_values is None:
            print("Error: rus_values are required for TIME-MoE model")
        else:
            print("Analyzing TIME-MoE model expert activations...")
            time_moe_save_dir = os.path.join(save_dir, "time_moe")
            plot_all_moe_layers_time_moe(time_moe_model, data_batch, rus_values, 
                                   modality_names, time_moe_save_dir, moe_num_synergy_experts)
    
    if baseline_model is not None:
        print("Analyzing baseline model expert activations...")
        baseline_save_dir = os.path.join(save_dir, "baseline")
        plot_all_moe_layers_baseline(baseline_model, data_batch, 
                                   modality_names, baseline_save_dir, moe_num_synergy_experts,
                                   seed, subject)
    
    print(f"Analysis complete. Plots saved to {save_dir}")
