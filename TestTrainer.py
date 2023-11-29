import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, PreTrainedTokenizerFast
from peft import get_peft_model, LoraConfig, TaskType

def load_and_adapt_model(model_name_or_path, peft_config):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    return model

def load_tokenizer_from_files(tokenizer_directory):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_directory, "tokenizer.json"))
    tokenizer.special_tokens_map = json.load(open(os.path.join(tokenizer_directory, "special_tokens_map.json")))
    tokenizer.config = json.load(open(os.path.join(tokenizer_directory, "tokenizer_config.json")))
    return tokenizer


def prepare_data_for_training(data_json, tokenizer, max_length=512):
    # Initialize lists to store encoded inputs and outputs
    inputs, outputs = [], []

    for item in data_json:
        # Combine instruction and input to form the model input
        model_input = f"{item['Instrucción']} {item['Entrada']}"

        # Output is the expected response
        model_output = item['Salida']

        # Tokenize and encode the input and output
        encoded_input = tokenizer.encode_plus(model_input, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoded_output = tokenizer.encode_plus(model_output, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

        # Add the encoded input and output to the respective lists
        inputs.append(encoded_input)
        outputs.append(encoded_output)

    return inputs, outputs

def train_model(model, inputs, outputs, epochs=3, batch_size=8, learning_rate=5e-5):
    # Combine inputs and outputs into a TensorDataset
    dataset = TensorDataset(torch.cat([input['input_ids'] for input in inputs], dim=0),
                            torch.cat([output['input_ids'] for output in outputs], dim=0))

    # Create a DataLoader for handling batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set the model to training mode
    model.train()

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Split the batch into input and output
            input_ids, labels = batch

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# PEFT Configuration for LRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

# Toy data for testing
data_json = [
    {"Instrucción": "¿Cuál es la palabra en Kichwa que corresponde a 'ABDOMEN' en español?", "Entrada": "ABDOMEN", "Salida": "Wiksa, itsa."},
    {"Instrucción": "Menciona la palabra en Kichwa que se asemeja a 'LLAMAR'.", "Entrada": "LLAMAR", "Salida": "Kayana, caparina, shamui nina."},
    {"Instrucción": "¿Qué término en español representa a 'RIKCHAK'?", "Entrada": "RIKCHAK", "Salida": "Idéntico, parecido."}
]

# Load path to model files
model_path = '/path-to-llama-2-files' 

# Load Tokenizer
tokenizer = load_tokenizer_from_files(model_path)

# Prepare the data
inputs, outputs = prepare_data_for_training(data_json, tokenizer)

# Load and adapt the model
model = load_and_adapt_model(model_path, peft_config)

# Train the model 
train_model(model, inputs, outputs)
