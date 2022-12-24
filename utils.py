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

def Encode_TextWithAttention(sentence,tokenizer,maxlen,padding_type='max_length',attention_mask_flag=True):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True, padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids'],encoded_dict['attention_mask']

def Encode_TextWithoutAttention(sentence,tokenizer,maxlen,padding_type='max_length',attention_mask_flag=False):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True, padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids']

def get_TokenizedTextWithAttentionMask(sentenceList, tokenizer):
    token_ids_list,attention_mask_list = [],[]
    for sentence in sentenceList:
        token_ids,attention_mask = Encode_TextWithAttention(sentence,tokenizer,MAX_LEN)
        token_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
    return token_ids_list,attention_mask_list

def get_TokenizedText(sentenceList, tokenizer):
    token_ids_list = []
    for sentence in sentenceList:
        token_ids = Encode_TextWithoutAttention(sentence,tokenizer,MAX_LEN)
        token_ids_list.append(token_ids)
    return token_ids_list

def get_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def mydata_loader(data_train, data_val, data_test, tokenizer):
    wordnet = WordNetLemmatizer()
    porter  = PorterStemmer()

    data_train['tweet'] = data_train.apply(lambda x: row_preprocess(x, wordnet, porter), 1)
    data_val['tweet'] = data_val.apply(lambda x: row_preprocess(x, wordnet, porter), 1)
    data_test['tweet'] = data_test.apply(lambda x: row_preprocess(x, wordnet, porter), 1)

    data_train['label_encoded'] = data_train.apply(lambda x: map_label(x), 1)
    data_val['label_encoded'] = data_val.apply(lambda x: map_label(x), 1)

    train_sentences = data_train.tweet.values
    val_sentences = data_val.tweet.values
    test_sentences = data_test.tweet.values

    train_labels = data_train.label_encoded.values
    val_labels = data_train.label_encoded.values

    train_token_ids,train_attention_masks = torch.tensor(get_TokenizedTextWithAttentionMask(train_sentences,tokenizer))
    val_token_ids,val_attention_masks = torch.tensor(get_TokenizedTextWithAttentionMask(val_sentences,tokenizer))
    test_token_ids,test_attention_masks = torch.tensor(get_TokenizedTextWithAttentionMask(test_sentences,tokenizer))

    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)


    train_data = TensorDataset(train_token_ids, train_attention_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=CONFIG_BATCH_SIZE)

    validation_data = TensorDataset(val_token_ids, val_attention_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=CONFIG_BATCH_SIZE)

    test_data = TensorDataset(test_token_ids, test_attention_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=CONFIG_BATCH_SIZE)

    return train_dataloader, validation_dataloader, test_dataloader



