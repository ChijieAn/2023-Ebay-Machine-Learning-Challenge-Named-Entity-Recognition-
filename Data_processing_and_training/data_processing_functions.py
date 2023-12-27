import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, TrainerCallback
import torch
from sklearn.metrics import f1_score

def turn_all_the_tag_to_string(df):
  for index,row in df.iterrows():
    if pd.isna(row['Tag']):
            print('Original value is NaN, converting to string "NaN".')
            df.at[index, 'Tag'] = 'NaN'
  return df

def update_tags(df):
    # Initialize a variable to keep track of the current group tag
    current_tag = None
    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # If the tag is not 'NaN', update the current tag and continue
        current_tag= row['Tag']
        # If the current tag is not None and the previous tag is not 'NaN', it means it's part of a group
        if current_tag == 'NaN' and df.loc[index - 1, 'Tag'] != 'NaN':
            # Update the tag to the current tag with the '-in' suffix
            if df.loc[index - 1, 'Tag'][-3:]!='-in':
              df.at[index, 'Tag'] = df.loc[index - 1, 'Tag'] + '-in'
            else:
              df.at[index, 'Tag'] = df.loc[index - 1, 'Tag']

    return df

#the class to process the dataset, apply masks to the dataset
class NerDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, idx):

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)
    
#deal with the problem of the subwords are genearted through the tokenizer
#we give the subwords genearted from the same word the same token corresponding to the token of the word
def align_labels_with_subwords2(tokenized_texts, sentences):
    aligned_labels = []

    for tokens, sentence in zip(tokenized_texts, sentences):
        labels = []
        word_index = 0
        is_prev_token_hyphenated = False

        print('the length of the sentence is ',len(sentence))
        print(tokens)
        print(sentence)

        for i, token in enumerate(tokens):
            # Check if the current token is a period or hyphen
            print('the index of the word is',word_index)
            print('the token is ',token)
            if token in ['.', '-'] and i > 0:
                # Check if the previous full token includes a period or hyphen
                prev_full_token = sentence[word_index][0] if word_index < len(sentence) else ''
                if '.' in prev_full_token or '-' in prev_full_token:
                    # Do not increment word index and assign the same label as the previous token
                    labels.append(labels[-1])
                    continue

            # Check if the token is the start of a new word
            if not token.startswith('##') and token not in ['.', '-']:
                # Update the word label only if it's a new word
                if word_index < len(sentence):
                    word_label = sentence[word_index][1]
                    labels.append(word_label)
                    word_index += 1
                else:
                    # If there are no more words in the sentence, use the last word's label
                    labels.append(labels[-1])
            else:
                # For subword tokens or '.', '-' tokens that are part of the previous word, use the previous label
                labels.append(labels[-1])

            # Check if the current token is a hyphenated part
            print(sentence[word_index-1][0])
            is_prev_token_hyphenated = token.startswith('##') and '-' in sentence[word_index - 1][0] if word_index > 0 else False

        aligned_labels.append(labels)

    return aligned_labels
