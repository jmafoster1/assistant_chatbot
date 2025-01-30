#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:49:45 2025

@author: lesya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 17:22:23 2023

@author: lesya
"""

#trains multi-lingual bert from preprocessed data 
#(uses 'train' dataset, splits it into train and validation. Will use the organiser dev set for final test))

#good for custom loss functions: https://lajavaness.medium.com/multiclass-and-multilabel-text-classification-in-one-bert-model-95c54aab59dc

import pandas as pd 
import torch 
import re
import csv
from datasets import Dataset, DatasetDict
# from skmultilearn.model_selection import iterative_train_test_split 
from sklearn.model_selection import train_test_split 
from functools import partial
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast, BertTokenizer, XLMRobertaTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, AutoTokenizer  
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

#paths
PATHTOTRAIN = './intent.tsv'


#train/dev split 
VALID_SIZE = 0

#model details
# BASE_MODEL = "bert-base-multilingual-cased"
BASE_MODEL="xlm-roberta-base"
LEARNING_RATE  = 1e-4
MAX_LENGTH = 512 
BATCH_SIZE =16
EPOCHS = 30
loss_fn = torch.nn.MSELoss()


if torch.cuda.is_available():      
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


#load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
tokenizer = XLMRobertaTokenizer.from_pretrained(BASE_MODEL)

# model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=6)
model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=6)

 

# 
names_all = ['question','greeting', 'feedback_bug',	'feedback_feature',	'feedback_output',	'explanation',	'end',	'lang']


#load data 
df_train = pd.read_csv(PATHTOTRAIN, skiprows=1,
                          encoding = "utf-8", names=names_all, delimiter='\t')


training_set = df_train.sample(frac=1).reset_index(drop=True)
training_set.greeting=training_set.greeting.astype(float)
training_set.feedback_bug=training_set.feedback_bug.astype(float)
training_set.feedback_feature=training_set.feedback_feature.astype(float)
training_set.feedback_output=training_set.feedback_output.astype(float)
training_set.explanation=training_set.explanation.astype(float)
training_set.end=training_set.end.astype(float)
training_set["labels"]=training_set[['greeting', 'feedback_bug',	'feedback_feature',	'feedback_output',	'explanation',	'end']].values.tolist()


# le = preprocessing.LabelEncoder()
# le.fit(training_set.label.values)
# training_set.label=le.transform(training_set.label)



def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    text=text.strip()
    sents=tokenizer.tokenize(text)
    start=""
    end=""
    if len(sents)>512:
        sentence=re.split('(?<=[\.\?\!])\s*', text)
        # start eppending sentences to the end and start
        for count,sent in enumerate(sentence):
            if len(tokenizer.tokenize(start+" "+end))<512:
                start=start+" "+sent
                if len(sentence)-count-1<count:
                    end=end+str(sentence[len(sentence)-count-1:len(sentence)-count])
        text=start+" "+end
    return text


def encode_batch(tokenizer, batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["question"], truncation=True, padding="max_length")


## For fine-tuning Bert, the authors recommmend a batch size of 16 or 32

def make_dataset(tokenizer: AutoTokenizer, df, split=True, test_size=0.1) -> Dataset:
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(partial(encode_batch, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    if split:
        dataset = dataset.train_test_split(test_size=test_size, seed=2023)
    return dataset

#convert to huggingface datasets
data_train_test=make_dataset(tokenizer, training_set, split=True, test_size=0.1)
print(data_train_test["test"]["labels"])


def get_preds_from_logits(logits, threshold=0.58):
    probs = torch.nn.Sigmoid()(torch.from_numpy(logits))
    print(probs)
    preds = torch.zeros(probs.shape)
    preds[torch.where(probs >= threshold)] = 1
    # probs = torch.argmax(torch.sigmoid(logits), dim=1).flatten()
    return preds



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(logits)
    print(labels)
    final_metrics = {}
    
    # Deduce predictions from logits
    predictions = get_preds_from_logits(logits)
    print(predictions)

    # The global f1_metrics
    final_metrics["f1_micro"] = f1_score(labels, predictions, average="micro")
    final_metrics["f1_macro"] = f1_score(labels, predictions, average="macro")
    
    # Classification report
    print("Classification report for global scores: ")
    print(classification_report(labels, predictions, zero_division=0))
    return final_metrics


class MyTrainer(Trainer):
    # def __init__(self, group_weights=None, **kwargs):
    #     super().__init__(**kwargs)
    #     self.group_weights = group_weights #to experiment with - weighting imbalanced classes
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        print
        outputs = model(**inputs)
        logits = outputs[0]
        # print(logits)
        # print(labels)
        loss = loss_fn(logits, labels)

        # loss = self.group_weights * loss 
        return (loss, outputs) if return_outputs else loss


class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")


training_args = TrainingArguments(
    output_dir="./models/bert",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,


    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",


    save_total_limit=1,
    metric_for_best_model="f1_macro",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=data_train_test["train"],
    eval_dataset=data_train_test["test"],
    compute_metrics=compute_metrics,
    callbacks=[PrinterCallback])

trainer.train(                                                )
trainer.save_model("./models/last")

