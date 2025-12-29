import os
import sys
from dotenv import load_dotenv
from meganova import MegaNova, MeganovaError
import time

load_dotenv()
api_key = os.getenv("MEGANOVA_API_KEY")

if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    print("Please set MEGANOVA_API_KEY in your .env file.")
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
                return None
    except Exception as e:
        print(f"Error fetching models: {e}")
    return None

def test_context_length(model_id: str, start_tokens=1000, max_target=None):
    print(f"\nTesting context length for model: {model_id}")
    
    if max_target is None:
        claimed = get_model_context_length(model_id)
        if claimed:
            print(f"Model claims context length: {claimed}")
            max_target = claimed
        else:
            print("Could not determine claimed context length. Defaulting to 128k check.")
            max_target = 132000 # Go slightly above 128k
            
    # Binary search setup
    low = start_tokens
    high = max_target + 1000 # overshoot slightly
    max_success = 0
    
    # We use a simple word "the " which is 4 chars. 
    # This is reliably 1 token in many tokenizers.
    token_word = "the "
    
    print(f"Starting binary search between {low} and {high} estimated tokens...")
    
    while low <= high:
        mid = (low + high) // 2
        
        # Construct prompt
        prompt = token_word * mid
        
        try:
            print(f"Testing approx {mid} tokens...", end="", flush=True)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=10, # Small max_tokens for quick response, not relevant for context test
                temperature=0.0,
            )
            # If successful, get the actual used tokens
            actual_tokens = response.usage.prompt_tokens
            print(f" Success! (Used: {actual_tokens})")
            
            max_success = max(max_success, actual_tokens)
            low = mid + 1 # Move up
            time.sleep(0.1) # Small delay to avoid rate limits
            
        except MeganovaError as e:
            err_str = str(e).lower()
            if "context" in err_str or "length" in err_str or "too long" in err_str or "400" in err_str or "500" in err_str or "internal server" in err_str:
                print(f" Failed. (Context limit reached or capacity error)")
                print(f"  Error details: {e}")
                high = mid - 1 # Move down
            else:
                print(f" Failed with unexpected error: {e}")
                break
        except Exception as e:
            print(f" Failed with unexpected error: {e}")
            break
            
    print(f"\n--- Result ---")
    print(f"Model: {model_id}")
    print(f"Max confirmed context: {max_success} tokens")

if __name__ == "__main__":
    target_model = "meganova-ai/manta-flash-1.0"
    if len(sys.argv) > 1:
        target_model = sys.argv[1]
    
    test_context_length(target_model)
