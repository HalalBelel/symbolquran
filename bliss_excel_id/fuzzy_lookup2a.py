import csv
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd # Using pandas for easier CSV handling and NaN checks

# Configuration
input_filename = 'input.csv'
output_filename = 'output.csv'
# Using a model good for semantic similarity of short phrases/words
# Other options: 'all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2' (if multilingual needed)
model_name = 'all-MiniLM-L6-v2' 

print(f"Loading sentence transformer model: {model_name}...")
# Load the pre-trained Sentence Transformer model
# Use try-except to handle potential download issues or model not found
try:
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model '{model_name}': {e}")
    print("Please ensure the model name is correct and you have an internet connection")
    print("if the model needs to be downloaded.")
    exit()

# --- Step 1: Read input CSV and build synonym dictionary ---
print(f"Reading input CSV: {input_filename}...")
all_synonyms = []
all_indices = []
input_data = []

try:
    # Use pandas to read CSV, handles potential encoding issues and NaNs better
    df_input = pd.read_csv(input_filename, encoding='utf-8-sig') # utf-8-sig handles potential BOM
    
    # Convert dataframe to list of dictionaries for easier processing
    input_data = df_input.to_dict('records')

    print("Building synonym dictionary...")
    for i, row in enumerate(input_data):
        index_num_str = str(row.get('Index Number', '')).strip()
        word_str = str(row.get('Word', '')).strip() # Ensure 'Word' is string and handle NaN

        if not index_num_str:
             print(f"Warning: Row {i+2} has missing Index Number. Skipping.")
             continue
             
        if not word_str:
             # Allow rows with empty 'Word' field, just don't add synonyms
             continue

        # Split words by comma and clean up
        synonyms = [syn.strip() for syn in word_str.split(',') if syn.strip()]
        
        for synonym in synonyms:
            all_synonyms.append(synonym)
            # Store index number as string for easier joining later
            all_indices.append(index_num_str) 

    if not all_synonyms:
        print("Warning: No synonyms found in the 'Word' column. Output might be incomplete.")
    else:
        print(f"Built dictionary with {len(all_synonyms)} synonyms.")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
    exit()
except Exception as e:
    print(f"Error reading or processing input CSV: {e}")
    exit()

# --- Step 2: Pre-compute embeddings for all synonyms ---
# Only proceed if there are synonyms to embed
synonym_embeddings = None
if all_synonyms:
    print("Computing embeddings for synonyms...")
    try:
        synonym_embeddings = model.encode(all_synonyms, convert_to_tensor=True, show_progress_bar=True)
        print("Synonym embeddings computed.")
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        exit()
else:
    print("Skipping embedding computation as no synonyms were found.")


# --- Step 3: Process each row and perform matching ---
print("Processing rows and performing semantic matching...")
output_data = []

for i, row in enumerate(input_data):
    print(f"Processing row {i+1}/{len(input_data)}...")
    breakdown_str = str(row.get('Breakdown of word', '')).strip() # Ensure string and handle NaN
    
    matched_words_list = []
    matched_indices_list = []

    # Only process if breakdown string is not empty and we have synonyms to match against
    if breakdown_str and all_synonyms and synonym_embeddings is not None:
        # Remove outer parentheses
        # Using strip('()') handles cases like '(word + word)' but not nested or complex structures.
        # Based on examples, this seems the likely intended simple approach.
        cleaned_breakdown = breakdown_str.strip('()')
        
        # Split by '+' and clean tokens
        tokens = [token.strip() for token in cleaned_breakdown.split('+') if token.strip()]

        if tokens:
            try:
                # Compute embeddings for the current row's tokens
                token_embeddings = model.encode(tokens, convert_to_tensor=True)

                # Use semantic_search to find the best match for each token
                # top_k=1 gives us the single best match for each token query
                hits = util.semantic_search(token_embeddings, synonym_embeddings, top_k=1)

                # hits is a list of lists (one list per token query)
                for token_hits in hits:
                    if token_hits: # Check if any match was found for this token
                        best_match_info = token_hits[0]
                        corpus_id = best_match_info['corpus_id'] # Index in the all_synonyms list
                        
                        matched_words_list.append(all_synonyms[corpus_id])
                        matched_indices_list.append(all_indices[corpus_id])
                    else:
                        # Handle cases where no match is found (optional, semantic_search usually finds something)
                        # matched_words_list.append("NO_MATCH") 
                        # matched_indices_list.append("NO_MATCH_IDX")
                        # For this task, we'll assume a match is always found or skip it
                        pass 
            except Exception as e:
                print(f"Error during semantic search for row {i+1}: {e}")
                # Optionally add error placeholders
                # matched_words_list = ["ERROR"] * len(tokens)
                # matched_indices_list = ["ERROR"] * len(tokens)
                pass # Continue processing other rows

    # Add original data and the new matched data to the output row
    output_row = row.copy() # Start with original data
    output_row['Matched breakdown of word'] = ' + '.join(matched_words_list)
    output_row['Matched breakdown of word index number'] = ' + '.join(matched_indices_list)
    output_data.append(output_row)

# --- Step 4: Write output CSV ---
print(f"Writing output CSV: {output_filename}...")

# Define the exact order of columns for the output file
# Ensure all original columns are included plus the new ones
if output_data:
    # Dynamically get original fieldnames from the first processed row
    # This handles cases where the input might have slightly different casing or extra columns
    original_fieldnames = list(output_data[0].keys())
    # Remove the new columns if they accidentally got added before defining fieldnames
    original_fieldnames.remove('Matched breakdown of word')
    original_fieldnames.remove('Matched breakdown of word index number')
    
    # Define the final fieldnames order
    fieldnames = original_fieldnames + ['Matched breakdown of word', 'Matched breakdown of word index number']
else:
    # Fallback if input was empty or processing failed early
    fieldnames = ['Index Number', 'Word', 'Breakdown of word', 'Matched breakdown of word', 'Matched breakdown of word index number']
    print("Warning: No data to write to output file.")


try:
    with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_data)
    print("Output CSV file created successfully.")
except Exception as e:
    print(f"Error writing output CSV: {e}")

print("Processing finished.")
