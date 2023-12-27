#load the data provided by eBay. Due to copy right, the original datafile is deleted after the competition ends
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, TrainerCallback
import torch
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences

from data_processing_functions import (turn_all_the_tag_to_string,update_tags,
                                       NerDataset,align_labels_with_subwords2)

from training_functions import (compute_metrics2,BertCRF,F1ScoreCallback)

data = pd.read_csv('/content/Train_Data_Fixed.tsv', delimiter='\t', encoding='utf-8', error_bad_lines=False)

#if the tag is Nah, convert it to string Nah

data = turn_all_the_tag_to_string(data)


#update the tags, use BOI tagging strategy to deal with the words within phrases
data=update_tags(data)

#import the pretrained bert tokenizer
MAX_LEN = 75
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

#pad each sentence to make the sentences of the same length
words = list(set(data["Token"].values))
words.append("ENDPAD")
tags = sorted(list(set(data["Tag"].values)))


word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

agg_func = lambda s: [(w, t) for w, t in zip(s["Token"].values.tolist(), s["Tag"].values.tolist())]
grouped = data.groupby("RN").apply(agg_func)
sentences = [s for s in grouped]

tokenized_texts = [tokenizer.tokenize(" ".join([str(s[0]) for s in sent])) for sent in sentences]
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post")

#this is the wrong version of labels and it can't match with the tokenized tokens
#labels = [[s[1] for s in sent] for sent in sentences]

#get the aligned labels which can match well with the tokenized tokens
aligned_labels=align_labels_with_subwords2(tokenized_texts,sentences)

tags_padded = pad_sequences([[tag2idx[l] for l in lab] for lab in aligned_labels],
                            maxlen=MAX_LEN, value=tag2idx["No Tag"], padding="post",
                            dtype="long", truncating="post")

tags_flat = np.array(tags_padded).flatten()\

#add the attention mask to the model
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

#split the train and validation dataset
train_indices, val_indices, train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    range(len(sentences)),
    input_ids,
    tags_padded,
    test_size=0.2,
    random_state=42
)

#also split the attention mask using the same method
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.2)

#convert the dataframe to torch tensor
tr_inputs = torch.tensor(train_sentences)
val_inputs = torch.tensor(val_sentences)
tr_tags = torch.tensor(train_labels)
val_tags = torch.tensor(val_labels)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_dataset = NerDataset(tr_inputs, tr_masks, tr_tags)
eval_dataset = NerDataset(val_inputs, val_masks, val_tags)

#import the pretrained Bert german model for NER task
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, TrainerCallback
# Your BERT model code
model = BertForTokenClassification.from_pretrained(
    "dbmdz/bert-base-german-cased",
    output_attentions=False,
    output_hidden_states=False,
    num_labels=68
)
model.to('cuda')

bert_model_name="dbmdz/bert-base-german-cased"

#add a CRF layer to the original bert model
Bert_CRF=BertCRF(bert_model_name,num_labels=68)

#define the training arguments
training_args = TrainingArguments(
    output_dir=f'/content/drive/My Drive/Bert_CRF/',
    num_train_epochs=40,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=3e-5,
    gradient_accumulation_steps=2,
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",  # Change this to use weighted F1 as the best metric
)

#train the model
trainer = Trainer(
    model=Bert_CRF,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics2, # Add compute metrics function here
     callbacks=[F1ScoreCallback()]
)

trainer.train()