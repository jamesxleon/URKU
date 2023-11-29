import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def load_llama_model(model_dir):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Initialize an empty model
    model = AutoModelForCausalLM.from_pretrained(None, config=os.path.join(model_dir, "config.json"), return_dict=True)

    # Load model shards
    shard_files = sorted([f for f in os.listdir(model_dir) if f.startswith('pytorch_model-') and f.endswith('.bin')])
    state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(model_dir, shard)
        state_dict.update(torch.load(shard_path, map_location="cpu"))

    model.load_state_dict(state_dict)

    return model, tokenizer

def generate_text(model, tokenizer, input_text, max_length=50):
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text using the model
    output = model.generate(input_ids, max_length=max_length)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def chat_with_model(model, tokenizer):
    print("Start chatting with the model (type 'quit' to stop):")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = generate_text(model, tokenizer, user_input)
        print(f"Model: {response}")

model_dir = '/path-to-model' # This must be set to the LlaMA-2 folder with the .safetensors and .bin files 
model, tokenizer = load_llama_model(model_dir)
input_text = "Let's play with the model to check if it works!!!"
generated_text = generate_text(model, tokenizer, input_text)

print(generated_text)

# For Chat-like interactions
#chat_with_model(model, tokenizer)