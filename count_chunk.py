import pandas as pd
import pickle

# Load your metadata from a pickle file or call the chunk extraction function
# Example: metadata = pickle.load(open("metadata.pkl", "rb"))
# Or, call your loader:
# from document_loader import extract_text_chunks_with_metadata
# _, metadata = extract_text_chunks_with_metadata("path_to_pdfs")

# Here assuming metadata list of dicts exists in current runtime:
def count_chunks(metadata):
    df = pd.DataFrame(metadata)
    print("Chunks per PDF file:")
    print(df['source_file'].value_counts())

    print("\nChunks per PDF & Page:")
    print(df.groupby(['source_file', 'page']).size())

# Example usage:
if __name__ == "__main__":
    # Load metadata (uncomment and update as needed)
    # with open("metadata.pkl", "rb") as f:
    #     metadata = pickle.load(f)

    # Or if metadata is not saved, extract again here
    from document_loader import extract_text_chunks_with_metadata
    _, metadata = extract_text_chunks_with_metadata("data")
    
    count_chunks(metadata)
