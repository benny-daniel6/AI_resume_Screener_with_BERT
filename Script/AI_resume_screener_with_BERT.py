import pandas as pd
import numpy as np

df= pd.read_csv('Data/resumedataset.csv')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()
df["Category"] = label_encoder.fit_transform(df["Category"])
labelmap=dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
train_text,val_text,train_labels,val_labels=train_test_split(df['Resume'],df['Category'],test_size=0.2,random_state=42)

from transformers import BertTokenizer, BertForSequenceClassification,Trainer, TrainingArguments
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=len(labelmap))

trainencodings=tokenizer(list(train_text),truncation=True,padding=True,max_length=512)
valencodings=tokenizer(list(val_text),truncation=True,padding=True,max_length=512)

import torch
class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self,idx):
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)
traindataset=ResumeDataset(trainencodings,train_labels)
valdataset=ResumeDataset(valencodings,val_labels)

training_args=TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    eval_strategy="epoch",
                   save_strategy="epoch",               
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)
trainer=Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=traindataset,          # training dataset
    eval_dataset=valdataset              # evaluation dataset
)
trainer.train()
eval_results=trainer.evaluate()
print(eval_results)

import gradio as gr

def predict_resume_category(resume_text):
    inputs=tokenizer(resume_text,truncation=True,padding=True,max_length=512,return_tensors='pt')
    outputs=model(**inputs)
    probs=torch.nn.functional.softmax(outputs.logits,dim=1)
    predicted_class=torch.argmax(probs,dim=1).item()
    category=label_encoder.inverse_transform([predicted_class])[0]
    return category
interface=gr.Interface(fn=predict_resume_category,
                       inputs=gr.Textbox(lines=15,label="Enter Resume Text"),
                       outputs=gr.Textbox(label="Predicted Job Category"),
                       title="Resume Screener",
                       description="Enter a resume text to predict the job category.")
interface.launch()
torch.save(model.state_dict(), 'resume_screener_model.pth')