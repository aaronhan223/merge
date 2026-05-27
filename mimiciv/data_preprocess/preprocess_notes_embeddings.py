import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel



# Copied from HAIM's repo: https://github.com/lrsoenksen/HAIM/blob/main/MIMIC_IV_HAIM_API.py
def split_note_document(text, biobert_tokenizer, min_length = 15):
    # Inputs:
    #   text -> String of text to be processed into an embedding. BioBERT can only process a string with â‰¤ 512 tokens. If the 
    #           input text exceeds this token count, we split it based on line breaks (driven from the discharge summary syntax). 
    #   min_length ->  When parsing the text into its subsections, remove text strings below a minimum length. These are generally 
    #                  very short and encode minimal information (e.g. 'Name: ___'). 
    #
    # Outputs:
    #   chunk_parse -> A list of "chunks", i.e. text strings, that breaks up the original text into strings with â‰¤ 512 tokens
    #   chunk_length -> A list of the token counts for each "chunk"
  
    # %% EXAMPLE OF USE
    # chunk_parse, chunk_length = split_note_document(ext, biobert_tokenizer, min_length = 15)
  
    tokens_list_0 = biobert_tokenizer.tokenize(text)
  
    if len(tokens_list_0) <= 510:
        # return [text], [1]
        return [text], [len(tokens_list_0)]
    #print("Text exceeds 512 tokens - splitting into sections")
  
    chunk_parse = []
    chunk_length = []
    chunk = text
  
    ## Go through text and aggregate in groups up to 510 tokens (+ padding)
    tokens_list = biobert_tokenizer.tokenize(chunk)
    if len(tokens_list) >= 510:
        temp = chunk.split('\n')
        ind_start = 0
        len_sub = 0
        for i in range(len(temp)):
            temp_tk = biobert_tokenizer.tokenize(temp[i])
            if len_sub + len(temp_tk) >  510:
                chunk_parse.append(' '.join(temp[ind_start:i]))
                chunk_length.append(len_sub)
                # reset for next chunk
                ind_start = i
                len_sub = len(temp_tk)
            else: 
                len_sub += len(temp_tk)
    elif len(tokens_list) >= min_length:
        chunk_parse.append(chunk)
        chunk_length.append(len(tokens_list))
    #print("Parsed lengths: ", chunk_length)
      
    return chunk_parse, chunk_length

# Modified from HAIM's repo: https://github.com/lrsoenksen/HAIM/blob/main/MIMIC_IV_HAIM_API.py
def get_biobert_embeddings_batch(texts, biobert_tokenizer, biobert_model, device, batch_size=16):
    """
    Get BioBERT embeddings for a batch of texts.
    
    Args:
        texts: List of text strings
        batch_size: Number of texts to process at once
        
    Returns:
        List of embeddings, one for each input text
    """
    
    if not texts:
        return []
    
    # Move model to device if not already there
    device_obj = torch.device(device)
    if biobert_model.device != device_obj:
        biobert_model = biobert_model.to(device)
        print(f"Moved biobert_model to {device}")
    
    embeddings_list = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing BioBERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        tokens_pt = biobert_tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Move tokens to device
        tokens_pt = {key: value.to(device) for key, value in tokens_pt.items()}
        
        # Get embeddings
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = biobert_model(**tokens_pt)
            # Only keep pooler output, not hidden states to save memory
            batch_embeddings = outputs.pooler_output.detach().cpu().numpy()  # Move back to CPU for numpy
        
        embeddings_list.extend(batch_embeddings)
        
        # Clean up GPU memory after each batch
        del tokens_pt, outputs
        torch.cuda.empty_cache() if device.startswith('cuda') else None
    
    return embeddings_list

def process_notes_embeddings_batched(df, biobert_tokenizer, biobert_model, device, chunk_batch_size=8, note_batch_size=1000):
    """
    Process note embeddings in batches.
    
    Args:
        df: DataFrame with notes
        batch_size: Number of notes to process at once
        chunk_batch_size: Number of chunks to process in each BioBERT batch
        
    Returns:
        DataFrame with embeddings added
    """
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    result_df['biobert_embeddings'] = None
    
    # Collect all chunks and their metadata
    all_chunks = []
    chunk_metadata = []  # (row_index, chunk_index_within_note)
    
    print("Splitting notes into chunks...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        curr_text = row['text']
        chunk_parse, _ = split_note_document(curr_text, biobert_tokenizer, min_length=15)
        
        for chunk_idx, chunk in enumerate(chunk_parse):
            all_chunks.append(chunk)
            chunk_metadata.append((idx, chunk_idx))
    
    print(f"Total chunks to process: {len(all_chunks)}")
    
    # Process all chunks in batches
    print("Processing chunks with BioBERT...")
    all_embeddings = get_biobert_embeddings_batch(all_chunks, biobert_tokenizer, biobert_model, device, batch_size=chunk_batch_size)
    
    # Reorganize embeddings back to notes
    print("Reorganizing embeddings...")
    embeddings_by_note = {}
    
    for embedding, (row_idx, chunk_idx) in zip(all_embeddings, chunk_metadata):
        if row_idx not in embeddings_by_note:
            embeddings_by_note[row_idx] = []
        embeddings_by_note[row_idx].append(embedding)
    
    # Assign embeddings back to dataframe
    for row_idx, embeddings in embeddings_by_note.items():
        result_df.at[row_idx, 'biobert_embeddings'] = embeddings
    
    return result_df

def main(args):
    rad_notes_df = pd.read_parquet(os.path.join(args.output_dir, "rad_notes_text.parquet"))
    icu_rad_notes_df = rad_notes_df[rad_notes_df['stay_id'].notna()]

    biobert_tokenizer = AutoTokenizer.from_pretrained(args.biobert_path)
    biobert_model = AutoModel.from_pretrained(args.biobert_path)
    device = f'cuda:{args.device_number}' if args.device_number is not None else 'cpu'
    icu_rad_notes_df = process_notes_embeddings_batched(icu_rad_notes_df, biobert_tokenizer, biobert_model, device, args.chunk_batch_size)
    icu_rad_notes_df.to_parquet(os.path.join(args.output_dir, "rad_notes_text_embeddings.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--biobert_path", type=str, help='Path to BioBERT model, e.g. "dmis-lab/biobert-v1.1"', default='dmis-lab/biobert-v1.1')
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    parser.add_argument("--chunk_batch_size", type=int, help='Chunk batch size', default=16)
    parser.add_argument("--device_number", type=int, help='Device number', default=None)
    args = parser.parse_args()
    main(args)
