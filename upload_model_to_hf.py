'''python upload_model_to_hf.py <path_to_model>'''
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# This script uploads a trained model to the Hugging Face Hub.
# It is designed to be run from the command line.

def main():
    """Loads a trained model and tokenizer and pushes them to the Hugging Face Hub."""
    parser = argparse.ArgumentParser(description="Upload a fine-tuned Causal LM model to Hugging Face Hub")
    parser.add_argument("output_dir", type=str, help="Path to the saved model directory")
    args = parser.parse_args()

    # --- Load model and tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    
    repo_name = args.output_dir.split("/")[-1]

    # Push to Hugging Face Hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

if __name__ == "__main__":
    main()
