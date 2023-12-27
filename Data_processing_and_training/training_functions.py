from sklearn.metrics import f1_score
from transformers import TrainerCallback, IntervalStrategy
from typing import Dict, Union
import pandas as pd
import numpy as np

import torch
from torch import nn
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from torchcrf import CRF

# Define the compute metrics function
# Here we also save the data processed in the compute metrics function for later check
def compute_metrics2(p):
    true_labels = p.label_ids
    pred_labels = p.predictions
    #print('hi')
    #print(pred_labels)
    # Flatten the labels and predictions
    true_labels = [label for sublist in true_labels for label in sublist if label != -100]
    pred_labels = [label for sublist in pred_labels for label in sublist if label != -100]

    #convert the lists of true_labels and pred_labels to numpy arrays
    true_labels_array=np.array(true_labels)
    pred_labels_array=np.array(pred_labels)

    #convert the np array to pd dataframe

    df_true_labels=pd.DataFrame({'true labels':true_labels_array})
    csv_file_path1 = "/content/drive/MyDrive/Validation&Training_tokenized/CRF_validation_tags_true_tokenized.csv"
    df_true_labels.to_csv(csv_file_path1, index=False)

    df_pred_labels=pd.DataFrame({'pred labels':pred_labels_array})
    csv_file_path2 = "/content/drive/MyDrive/Validation&Training_tokenized/CRF_validation_tags_predict_tokenized.csv"
    df_pred_labels.to_csv(csv_file_path2, index=False)

    #write the true labels and pred labels to a csv file to see what exactly it is

    NON_LABEL_ID = 43
    mask = true_labels_array != NON_LABEL_ID

    filtered_true_labels = true_labels_array[mask]
    filtered_pred_labels = pred_labels_array[mask]

    df_filtered_true_labels=pd.DataFrame({'true labels':filtered_true_labels})
    csv_file_path3 = "/content/drive/MyDrive/Validation&Training_tokenized/CRF_validation_tags_true_masked_tokenized.csv"
    df_filtered_true_labels.to_csv(csv_file_path3, index=False)

    df_filtered_pred_labels=pd.DataFrame({'pred_labels':filtered_pred_labels})
    csv_file_path4 = "/content/drive/MyDrive/Validation&Training_tokenized/CRF_validation_tags_predict_masked_tokenized.csv"
    df_filtered_pred_labels.to_csv(csv_file_path4, index=False)

    # Calculate F1 for non 'NO_LABEL' labels
    weighted_f1 = f1_score(filtered_true_labels, filtered_pred_labels, average='weighted', labels=np.unique(pred_labels))
    return {'weighted_f1': weighted_f1}


#self define a CRF layer which will be added to the original Bert model
class BertCRF(nn.Module):
    def __init__(self, bert_model_name, num_labels):
      super(BertCRF, self).__init__()
      self.bert=BertForTokenClassification.from_pretrained(bert_model_name,num_labels=num_labels)
      #self.dropout = nn.Dropout(0.1)
      #self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
      self.crf = CRF(num_labels, batch_first=True)
    def forward(self, input_ids, attention_mask=None, labels=None):
      outputs = self.bert(input_ids, attention_mask=attention_mask)
      #print(outputs.size())
      outputs_tensor=outputs[0]
      #print(labels)
      #print(type(labels))
      loss = -self.crf(outputs_tensor, labels, mask=attention_mask.type(torch.uint8), reduction='mean')
      prediction=torch.tensor(self.crf.decode(outputs_tensor))
      #print(prediction)
      return {"loss": loss,"prediction": prediction}
    
class F1ScoreCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'eval_weighted_f1' in logs:
            print(f"Step {state.global_step}: Weighted F1 = {logs['eval_weighted_f1']:.4f}")



