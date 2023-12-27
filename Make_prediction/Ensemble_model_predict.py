from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch import nn
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.metrics import classification_report
from transformers import pipeline
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from Predictions_functions import (EnsembleModel,get_formatted_prediction)
from data_processing_functions import(turn_all_the_tag_to_string,update_tags,NerDataset)

#create the dictionaries which map the name of tags and index
data = pd.read_csv('/content/Train_Data_Fixed.tsv', delimiter='\t', encoding='utf-8', error_bad_lines=False)
data = turn_all_the_tag_to_string(data)
data=update_tags(data)
MAX_LEN = 75
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
words = list(set(data["Token"].values))
words.append("ENDPAD")
tags = sorted(list(set(data["Tag"].values)))
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}


# Load the model from the shared folder
model_fold_1_path = "/content/drive/My Drive/Model_Fold_1/checkpoint-4000"  # Path to the shared folder
model_fold_1_4000 = BertForTokenClassification.from_pretrained(model_fold_1_path)

model_fold_2_path = "/content/drive/My Drive/Model_Fold_2/checkpoint-4000"
model_fold_2_4000 = BertForTokenClassification.from_pretrained(model_fold_2_path)

model_fold_3_path = "/content/drive/My Drive/Model_Fold_3/checkpoint-4000"
model_fold_3_4000 = BertForTokenClassification.from_pretrained(model_fold_3_path)

model_fold_4_path = "/content/drive/My Drive/Model_Fold_4/checkpoint-4000"
model_fold_4_4000 = BertForTokenClassification.from_pretrained(model_fold_4_path)

model_fold_5_path = "/content/drive/My Drive/Model_Fold_5/checkpoint-4000"
model_fold_5_4000 = BertForTokenClassification.from_pretrained(model_fold_5_path)

model_name = "dbmdz/bert-base-german-cased"

tokenizer = BertTokenizer.from_pretrained(model_name)

model_list=[model_fold_1_4000,model_fold_2_4000,model_fold_3_4000,model_fold_4_4000,model_fold_5_4000]

ensemble_model=EnsembleModel(model_list)
ensemble_model.to('cuda')

trainer_3 = Trainer(
    model=ensemble_model,
)

true_path= "/content/drive/MyDrive/Validation&Training_tokenized/validation_tags_true_masked_tokenized.csv"
prediction_path="/content/drive/MyDrive/Validation&Training_tokenized/validation_tags_predict_masked_tokenized.csv"

true_labels=pd.read_csv(true_path)
predicted_labels=pd.read_csv(prediction_path)

true_labels_lst=true_labels.values.tolist()
predicted_labels_lst=predicted_labels.values.tolist()

report = classification_report(true_labels_lst, predicted_labels_lst, output_dict=True)
print(report)

print("Weighted F1 score:", report['weighted avg']['f1-score'])

#We can test our prediction on the first 5000 rows of the listing titles
listing_data = pd.read_csv('/content/Listing_Titles.tsv', delimiter='\t', encoding='utf-8', error_bad_lines=False)
train_validation_data_tokens = listing_data.iloc[0:5000]["Title"].tolist()

tokenizer_texts_train_validation=[tokenizer.tokenize(t) for t in train_validation_data_tokens]
input_ids_train_validation = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenizer_texts_train_validation],
                               maxlen=MAX_LEN, dtype="long", value=0.0,
                               truncating="post", padding="post")

attention_masks_test = [[float(i != 0.0) for i in ii] for ii in input_ids_train_validation]

input_ids_train_validation = torch.tensor(input_ids_train_validation)
attention_masks_train_validation = torch.tensor(attention_masks_test)

dummy_label_train_validation = torch.zeros_like(input_ids_train_validation)

train_validation_dataset=NerDataset(input_ids_train_validation,attention_masks_train_validation,dummy_label_train_validation)
predictions_train_validation, _, _ = trainer_3.predict(train_validation_dataset)

formatted_predictions_train_validation = []

records_train_validation = listing_data.iloc[0:5000]["Record Number"].tolist()

formatted_prediction_train_validation, word_labels_train_validation=get_formatted_prediction(records_train_validation,tokenizer_texts_train_validation,predictions_train_validation,train_validation_data_tokens,idx2tag,tag2idx)

#then we evaluate the f1 score on the whole train and alidation dataset with the results we get from predictiny by our model
sequence_train_validation=[]
for index in range(5000):
    # Filter the dataframe to get rows with the current index
    sentence_df = data[data['RN'] == index+1]

    # Get the tags, convert them to indices using tag2idx, and append to sequences
    tag_sequence = [tag2idx.get(tag, -1) for tag in sentence_df['Tag'].tolist()]
    sequence_train_validation.append(tag_sequence)

#run the report method to check the f1 score o whole words in the train and validation dataset using the method we generate the output file
flattened_predictions_t_v = []
flattened_ground_truth_t_v = []
count=0
for pred, true in zip(word_labels_train_validation, sequence_train_validation):
    # Truncate predictions to the length of the ground truth for each sentence
    #print('length of prediction',len(pred))
    #print('length of true prediction',len(true))
    if len(flattened_predictions_t_v)!=len(flattened_ground_truth_t_v):
      break
    if len(pred)>len(true):
      truncated_pred = pred[:len(true)]

    #print('length of truncated prediction',len(truncated_pred))
      flattened_predictions_t_v.extend(truncated_pred)
      flattened_ground_truth_t_v.extend(true)
    elif len(pred)<len(true):
      truncated_ture=true[:len(pred)]

      flattened_predictions_t_v.extend(pred)
      flattened_ground_truth_t_v.extend(truncated_ture)

    else:
      flattened_predictions_t_v.extend(pred)
      flattened_ground_truth_t_v.extend(true)
    count+=1

# Compute weighted F1 score and classification report
print(len(flattened_ground_truth_t_v))
print(len(flattened_predictions_t_v))
report = classification_report(flattened_ground_truth_t_v, flattened_predictions_t_v, output_dict=True )

print(report)

df_formatted_predictions_test = pd.DataFrame(formatted_prediction_train_validation, columns=['Record Number', 'Aspect Name', 'Aspect Value'])
csv_file_path = "/content/drive/MyDrive/Validation&Training/prediction_trai_validation_4000_425.csv"
df_formatted_predictions_test.to_csv(csv_file_path, sep="\t",index=False , encoding="utf-8")

#Then we use the model to predict and generate the data required for submission
listing_data = pd.read_csv('/content/Listing_Titles.tsv', delimiter='\t', encoding='utf-8', error_bad_lines=False)
quiz_data_tokens = listing_data.iloc[5000:30000]["Title"].tolist()

tokenized_texts_test = [tokenizer.tokenize(t) for t in quiz_data_tokens]
input_ids_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_test],
                               maxlen=MAX_LEN, dtype="long", value=0.0,
                               truncating="post", padding="post")

attention_masks_test = [[float(i != 0.0) for i in ii] for ii in input_ids_test]

input_ids_test = torch.tensor(input_ids_test)
attention_masks_test = torch.tensor(attention_masks_test)

dummy_labels = torch.zeros_like(input_ids_test)

test_dataset = NerDataset(input_ids_test, attention_masks_test, dummy_labels)

predictions, _, _ = trainer_3.predict(test_dataset)

formatted_predictions = []

records = listing_data.iloc[5000:30000]["Record Number"].tolist()

formatted_prediction_test,_=get_formatted_prediction(records,tokenized_texts_test,predictions,quiz_data_tokens,idx2tag,tag2idx)

df_formatted_predictions_test = pd.DataFrame(formatted_prediction_test, columns=['Record Number', 'Aspect Name', 'Aspect Value'])
df_formatted_predictions_test = df_formatted_predictions_test.sort_values(by=['Record Number', 'Aspect Name'])
csv_file_path = "/content/drive/MyDrive/Validation&Training/prediction_test_4000_2129.tsv"
df_formatted_predictions_test.to_csv(csv_file_path, sep="\t",index=False , encoding="utf-8",header=False)
