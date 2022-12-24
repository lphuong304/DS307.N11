import preprocessor as p
import nltk
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
stop_words = set(stopwords.words('english'))
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from config import *
import numpy as np

def check_and_make_dir(path):
    if not os.exists(path):
        os.makedirs(path)

p.set_options(p.OPT.URL, p.OPT.EMOJI)

def row_preprocess(row, lemmatizer, stemmer):
    text = row['tweet']
    # text = text.strip('\xa0')
    text = p.clean(text)
    tokenization = nltk.word_tokenize(text)     
    tokenization = [w for w in tokenization if not w in stop_words]
    #   text = ' '.join([porter_stemmer.stem(w) for w in tokenization])
    #   text = ' '.join([lemmatizer.lemmatize(w) for w in tokenization])
    # text = re.sub(r'\([0-9]+\)', '', text).strip()    
    return text

def map_label(row):
    return 0 if row['label']=='real' else 1

def encodeAttention(sentence,tokenizer,maxlen,padding_type='max_length',attention_mask_flag=True):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True, padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids'],encoded_dict['attention_mask']

def getAttentionMask(sentenceList, tokenizer):
    token_ids_list,attention_mask_list = [],[]
    for sentence in sentenceList:
        token_ids,attention_mask = encodeAttention(sentence,tokenizer,MAX_LEN)
        token_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
    return token_ids_list,attention_mask_list


def get_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)




