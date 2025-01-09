from transformers import BertForSequenceClassification, BertTokenizer
import random
import torch
import json


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder))
model.load_state_dict(torch.load('Shanks.pth'))
model.to(device)


def chat():
    while True:
        user_input = input("You: ")
        if user_input == "quit":
            break

        inputs = tokenizer(user_input, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()

        # Find the corresponding intent in the data list
        for intent in data['intents']:
            if intent['intent'] == list(label_encoder.keys())[predicted_class_id]:
                response = random.choice(intent['responses'])
                break

        print(f"Shanks: {response}")