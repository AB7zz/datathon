from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd




def main():
    # Step 1: Load Data
    data = pd.read_csv("sample_data.csv")

    # Step 2: Extract Textual Data
    # Assuming your data contains columns with textual descriptions
    textual_data = ' '.join(data.iloc[:10]["Description"])
    # print(textual_data)
    # Add your API token here
    api_token = "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"

    # Set up the authentication token
    hf_config = {"use_auth_token": api_token}
    # hf_config = {"token": api_token}

    # Load the LLAMA 2 model for text classification
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **hf_config)

    # Sample input text
    text = "I really enjoyed this movie. The plot was engaging and the acting was superb."
    # text = textual_data
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Forward pass through the model
    outputs = model(**inputs)

    # Get predicted class probabilities
    probs = torch.softmax(outputs.logits, dim=-1)

    # Get predicted label
    predicted_class = torch.argmax(probs, dim=-1).item()

    # Get label names
    label_names = tokenizer.model_input_names

    print("Predicted Class:", label_names[predicted_class])
    print("Predicted Class Probabilities:", probs)

if __name__ == "__main__":
    main()
