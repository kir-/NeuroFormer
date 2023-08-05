import torch
from gpt2_module import GPT2Module
from transformers import GPT2Tokenizer

def generate_response(model, text, max_length=100):
    # Tokenize input
    input_ids = model.tokenizer.encode(text, return_tensors='pt').to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.model.generate(input_ids, max_length=max_length, temperature=0.7, pad_token_id=model.tokenizer.eos_token_id)
    
    # Decode and return the output
    return model.tokenizer.decode(output[0], skip_special_tokens=True)

# Load trained model
checkpoint_path = "model.ckpt"
model = GPT2Module.load_from_checkpoint(checkpoint_path)
model.eval()
model.to("cuda:0" if torch.cuda.is_available() else "cpu")

# Demonstration
demo_text = "Once upon a time"
print(generate_response(model, demo_text))
