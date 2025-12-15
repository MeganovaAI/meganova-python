import os
import sys
from dotenv import load_dotenv
from meganova import MegaNova

# Load environment variables
load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    exit(1)

client = MegaNova(api_key=api_key)

def get_model_context_length(model_id: str) -> int:
    """Fetch the claimed context length from the API."""
    try:
        models = client.models.list()
        for model in models:
            if model.id == model_id:
                if model.context_length:
                    return model.context_length
    except Exception as e:
        print(f"Error fetching models: {e}")
    return None

def test_context_length(model_id: str):
    print(f"\nTesting context length for model: {model_id}")
    
    claimed_length = get_model_context_length(model_id)
    if claimed_length:
        print(f"Model claims context length: {claimed_length}")
    else:
        print("Model does not specify context length in metadata.")

    # Standard context windows to test
    milestones = [
        4096,       # 4k
        8192,       # 8k
        16384,      # 16k
        32768,      # 32k
        65536,      # 64k
        131072,     # 128k
        262144,     # 256k (approx)
    ]
    
    # We use "the " which is ~1 token in many tokenizers
    token_word = "the " 
    max_success = 0
    
    for milestone in milestones:
        # If the model claims a lower limit, we can warn but let's test anyway 
        # to see if it allows more, or stop if it's way beyond.
        # But usually we just test up to the milestone.
        
        # Subtract a safety buffer for system prompts / framing
        target_tokens = int(milestone * 0.95) # use 95% of the limit
        
        print(f"Testing milestone {milestone} (sending approx {target_tokens} tokens)...", end="", flush=True)
        
        prompt = token_word * target_tokens
        
        try:
            # We use max_tokens=10 because we only care if the Input is accepted
            response = client.chat.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=10 
            )
            
            used_tokens = response.usage.prompt_tokens
            print(f" Success! (Used: {used_tokens})")
            max_success = used_tokens
            
        except Exception as e:
            print(f" Failed.")
            print(f"  Error: {e}")
            # If we fail at a milestone, we assume higher ones will also fail
            break

    print(f"\n--- Result ---")
    print(f"Model: {model_id}")
    print(f"Max confirmed context: {max_success} tokens")

if __name__ == "__main__":
    target_model = "meganova-ai/manta-flash-1.0"
    if len(sys.argv) > 1:
        target_model = sys.argv[1]
    
    test_context_length(target_model)
