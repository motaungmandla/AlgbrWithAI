model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder))
model.load_state_dict(torch.load('Shanks.pth'))
model.to(device)