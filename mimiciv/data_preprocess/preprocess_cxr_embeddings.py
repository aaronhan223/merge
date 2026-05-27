import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv
import skimage.io
import cv2

class MimicCXRDataset(Dataset):
    """Dataset class for MIMIC-CXR images."""
    
    def __init__(self, metadata_df, mimic_cxr_jpg_dir):
        self.metadata_df = metadata_df
        self.mimic_cxr_jpg_dir = mimic_cxr_jpg_dir
        
        # Pre-compute all file paths and filter valid ones
        self.valid_entries = []
        print("Pre-computing image paths...")
        for index, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df), desc="Checking paths"):
            curr_subject_id = int(row['subject_id'])
            curr_study_id = int(row['study_id'])
            curr_dicom_id = row['dicom_id']

            f_subfolder = "p" + str(curr_subject_id)[0:2]
            pt_folder = "p" + str(curr_subject_id)
            s_folder = "s" + str(curr_study_id)
            curr_f_path = os.path.join(mimic_cxr_jpg_dir, 'files', f_subfolder, pt_folder, s_folder, curr_dicom_id + ".jpg")
            
            if os.path.exists(curr_f_path):
                self.valid_entries.append({
                    'original_index': index,
                    'file_path': curr_f_path,
                    'subject_id': curr_subject_id,
                    'study_id': curr_study_id,
                    'dicom_id': curr_dicom_id
                })
        
        print(f"Found {len(self.valid_entries)} valid images out of {len(self.metadata_df)} total entries")
    
    def __len__(self):
        return len(self.valid_entries)
    
    def __getitem__(self, idx):
        entry = self.valid_entries[idx]
        
        try:
            # Load and preprocess image
            img = skimage.io.imread(entry['file_path'])
            img = xrv.datasets.normalize(img, 255)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)   
            img = img[None, :, :]  # Add channel dimension
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).float()
            
            return {
                'image': img_tensor,
                'original_index': entry['original_index'],
                'file_path': entry['file_path']
            }
        except Exception as e:
            print(f"Error loading {entry['file_path']}: {e}")
            # Return a dummy tensor if loading fails
            return {
                'image': torch.zeros((1, 224, 224), dtype=torch.float32),
                'original_index': entry['original_index'],
                'file_path': entry['file_path']
            }

def process_img_embeddings_batched(metadata_df, model, device, mimic_cxr_jpg_dir, batch_size=16, num_workers=4):
    """
    Process image embeddings in batches.
    The procedure follows HAIM (https://github.com/lrsoenksen/HAIM/blob/main/MIMIC_IV_HAIM_API.py) and FuseMOE (https://github.com/aaronhan223/FuseMoE/tree/main/src/preprocessing/mimiciv_preprocessing) but in a batched manner.
    
    Args:
        metadata_df: DataFrame with image metadata
        model: torchxrayvision model
        device: torch device (cuda or cpu)
        mimic_cxr_jpg_dir: Path to mimic-cxr-jpg directory
        batch_size: Number of images to process at once
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataFrame with embeddings added
    """
    
    # Create a copy to avoid modifying the original
    result_df = metadata_df.copy()
    result_df['densefeatures'] = None
    result_df['predictions'] = None
    
    # Create dataset and dataloader
    dataset = MimicCXRDataset(metadata_df, mimic_cxr_jpg_dir)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("No valid images found. Returning empty results.")
        return result_df
    
    # Use DataLoader with multiprocessing for efficient I/O
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.startswith('cuda') else False,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False  # Process all images, even if last batch is smaller
    )
    
    print("Processing images in batches with DataLoader...")
    
    # Process batches
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_images = batch['image'].to(device)
        # Convert tensor to numpy for indexing - handle both single item and batch
        if batch['original_index'].dim() == 0:  # Single item
            batch_indices = [batch['original_index'].item()]
        else:  # Batch
            batch_indices = batch['original_index'].numpy()
        
        # Process batch through model
        with torch.no_grad():
            # Extract dense features for the batch
            feats = model.features(batch_images)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            batch_densefeatures = feats.cpu().detach().numpy().reshape(len(batch_images), -1)
            
            # Get predictions for the batch
            preds = model(batch_images).cpu()
            batch_predictions = preds.detach().numpy()
        
        # Assign results back to dataframe
        for j, df_index in enumerate(batch_indices):
            result_df.at[df_index, 'densefeatures'] = batch_densefeatures[j]
            result_df.at[df_index, 'predictions'] = batch_predictions[j]
        
        # Clean up GPU memory after each batch
        del batch_images, feats, preds
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    return result_df

def main(args):
    print('Load metadata...')
    assert os.path.exists(os.path.join(args.output_dir, "cxr_metadata_with_time_delta.parquet")), "cxr_metadata_with_time_delta.parquet not found"
    metadata_df = pd.read_parquet(os.path.join(args.output_dir, "cxr_metadata_with_time_delta.parquet"))

    device = f'cuda:{args.device_number}' if args.device_number is not None else 'cpu'
    print(f'Using device: {device}')
    model_weights_name = 'densenet121-res224-chex'
    model = xrv.models.DenseNet(weights=model_weights_name)
    model = model.to(device)
    model.eval()  # Set to evaluation mode for inference
    
    print('Processing images to embeddings...')
    # Use DataLoader-based batched processing for efficient I/O
    metadata_df = process_img_embeddings_batched(
        metadata_df, model, device, args.mimic_cxr_jpg_dir, 
        args.batch_size, args.num_workers
    )
    
    print('Saving metadata...')
    metadata_df.to_parquet(os.path.join(args.output_dir, "mimic_cxr_embeddings.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_cxr_jpg_dir", type=str, required=True, help='Path to mimic-cxr-jpg v2.0.0 directory (e.g. mimic-cxr-jpg/2.0.0/)')
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    parser.add_argument("--batch_size", type=int, help='Batch size for processing images', default=64)
    parser.add_argument("--num_workers", type=int, help='Number of worker processes for data loading', default=4)
    parser.add_argument("--device_number", type=int, help='Device number', default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    


