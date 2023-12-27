import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertForTokenClassification
from collections import Counter

#build the ensemble model and make predictions on the validation dataset
class EnsembleModel(nn.Module):
    def __init__(self, model_list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        for model in self.models:
            model.eval()

    def forward(self, input_ids, attention_mask=None):
        avg_softmax = None
        for model in self.models:
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            softmax = F.softmax(logits, dim=-1)
            if avg_softmax is None:
                avg_softmax = softmax
            else:
                avg_softmax += softmax
        avg_softmax /= len(self.models)
        return torch.argmax(avg_softmax, dim=-1)

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
def decode_tokens(tokens):
  return tokenizer.convert_tokens_to_string(tokens)

def most_common_number(lst):
    counts = Counter(lst)
    max_count = max(counts.values())  
    most_common = [num for num, count in counts.items() if count == max_count]  
    most_common.sort(key=lambda x: lst.index(x))  
    return most_common[0]  

#define the function to generate the formatted prediction required for submission
def get_formatted_prediction(records,tokens,predictions,Titles,idx2tag,tag2idx):
  #create the list of Non tags
  counter=0

  lst_non_tag_words=['.','#','-','(',')','!!','[',']','/','...','*',',','****','--','(/','....']

  formatted_predictions_eval = []
  predictions_for_words=[]
  title_sentence = [sentence.split() for sentence in Titles]
  for records_eval, tokens_eval, preds_eval, title in zip(records, tokens, predictions, title_sentence):
      discrminant=False

      preds_word_labels=[]
      index=0
      index_title=0
      while index < len(tokens_eval) and index_title<len(title):
        preds_one_word=[preds_eval[index]]
        tokens_eval_pre_word=tokens_eval[index]
        if index+1 < len(tokens_eval) and tokens_eval[index]== '[UNK]':
          if tokens_eval[index+1] not in title[index_title]:
            index += 1
            index_title +=1
            result = most_common_number(preds_one_word)
            preds_word_labels.append(result)
            continue

        index += 1
        #print(index)


        while index < len(tokens_eval) and (tokens_eval[index].startswith("##") or tokens_eval_pre_word != title[index_title]):
          preds_one_word.append(preds_eval[index])
          if '[UNK]' in tokens_eval_pre_word :
            if tokens_eval[index].startswith("##"):
              tokens_eval_pre_word += tokens_eval[index].replace('##', '')
              if index+1 < len(tokens_eval) and tokens_eval[index+1] not in title[index_title]:
                index += 1
                break
            else:
              tokens_eval_pre_word += tokens_eval[index]
              if index+1 < len(tokens_eval) and not tokens_eval[index+1].startswith("##"):
                if tokens_eval[index+1] not in title[index_title]:
                  index += 1
                  break
          if tokens_eval[index].startswith("##"):
            tokens_eval_pre_word += tokens_eval[index].replace('##', '')
          elif tokens_eval[index]== '[UNK]':
            if index+1 < len(tokens_eval):
              if tokens_eval[index+1] in title[index_title]:
                tokens_eval_pre_word += tokens_eval[index]
              else:
                tokens_eval_pre_word += tokens_eval[index]
                index += 1
                break
          else:
            tokens_eval_pre_word += tokens_eval[index]
          index += 1
        index_title +=1
        result = most_common_number(preds_one_word)
        preds_word_labels.append(result)

      predictions_for_words.append(preds_word_labels)

      prev_label_id = None
      combined_token = ""
      if len(title)!=len(preds_word_labels) or discrminant==True:
        print('the record number is',records_eval)
        print('token length', len(tokens_eval))
        print(tokens_eval)
        print(preds_eval)
        print(tokens_eval_pre_word)
        print(title)
        print("preds_word_labels",preds_word_labels)
        counter+=1
        print(counter)
        #print(title)
      if len(title)!=len(preds_word_labels):
          continue
      for word, label_id in zip(title, preds_word_labels):
          #print(word,label_id)
          #print('this is the combined token',combined_token)
          if idx2tag[label_id] in ["No Tag", "Obscure","No Tag-in"] or word in lst_non_tag_words:
              #print('hi')
              #prev_label_id=label_id
              #print(prev_label_id)
              continue
          #print(idx2tag[label_id][:-3])
          if idx2tag[label_id][-3:]=='-in':
              if prev_label_id==None:
                prev_label_id=tag2idx[idx2tag[label_id][:-3]]
                combined_token=word
                continue
              elif idx2tag[label_id][:-3]==idx2tag[prev_label_id]:
                combined_token += " " + word
                new_id=idx2tag[label_id]
                new_id=new_id[:-3]
                label_id=tag2idx[new_id]
              else:
                #print('hi',word)
                if combined_token:
                  formatted_predictions_eval.append([records_eval, idx2tag[prev_label_id], combined_token])
                prev_label_id=tag2idx[idx2tag[label_id][:-3]]
                combined_token = word
                continue

          else:
              if combined_token:
                  formatted_predictions_eval.append([records_eval, idx2tag[prev_label_id], combined_token])
                  combined_token = ""
              combined_token = word



          prev_label_id = label_id

      if combined_token:
        formatted_predictions_eval.append([records_eval, idx2tag[prev_label_id], combined_token])

  return formatted_predictions_eval,predictions_for_words