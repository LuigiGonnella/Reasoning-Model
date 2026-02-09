import torch
from reasoning_from_scratch.qwen3 import download_qwen3_small
from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOKENIZER_PATH = Path("CH2/qwen3") / "tokenizer-base.json"
MODEL_PATH = Path('CH2/qwen3') / 'qwen3-0.6B-base.pth'

TOKENIZER = Qwen3Tokenizer(tokenizer_file_path=TOKENIZER_PATH)
MODEL = Qwen3Model(QWEN_CONFIG_06_B)
MODEL.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
MODEL.to(DEVICE)

#EXERCISE 2.2: STREAMING TOKEN GENERATION
#Write a modified version of the generate_text_basic function that returns each
#token as it is generated and prints it, which is also known as streaming token
#generation.

#The goal of this exercise is to understand how to implement token-by-token text
#generation, a technique often used in real-time applications like chatbots and
#interactive assistants.

@torch.inference_mode()
def generate_text_basic_stream(model, input_ids, max_new_tokens, eos_token_id=None):
    model.eval()

    for _ in range(max_new_tokens):
        out = model(input_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break

        yield next_token  # Yield each token as it's generated
        
        input_ids = torch.cat([input_ids, next_token], dim=1)

def extract_text(prompt, model, max_new_tokens, tokenizer, eos_token_id=None):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=DEVICE).unsqueeze(0)

    for token_id in generate_text_basic_stream(model, input_ids, max_new_tokens, eos_token_id):
        token_id = token_id.squeeze(0)
        print(tokenizer.decode([token_id]), end="", flush=True)
    
def main():
    prompt = "Explain why LeBron James is the GOAT"
    max_new_tokens = 150
    extract_text(prompt, MODEL, max_new_tokens, TOKENIZER, eos_token_id=TOKENIZER.eos_token_id)

if __name__ == "__main__":
    main()