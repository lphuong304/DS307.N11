import os
from utils import *
from config import *
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from transformers import AdamW
from transformers import XLMModel, BertTokenizer, BertForSequenceClassification
import preprocessor as p
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
import torch
import sys
sys.path.append('.')


stop_words = set(stopwords.words('english'))


def training(model, epochs, train_dataloader, validation_dataloader, optimizer):

    train_loss_set = []
    est_val_accuracy = 0.90

    for _ in trange(epochs, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                output = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
                logits = output[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = get_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    check_and_make_dir(CONFIG_SAVE_CHECKPOINTS)
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    Validation_Accuracy = (eval_accuracy/nb_eval_steps)
    if (Validation_Accuracy >= best_val_accuracy):
        torch.save(model.state_dict(), CONFIG_SAVE_CHECKPOINTS +
                   '/BERT_best_model.pt')
        best_val_accuracy = Validation_Accuracy
        print('Model Saved')


if __name__ == '__main__':
    device = CONFIG_DEVICE
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    data_train = pd.read_csv(os.path.join(CONFIG_PATH_INPUT, 'train.csv'))
    data_val = pd.read_csv(os.path.join(CONFIG_PATH_INPUT, 'val.csv'))
    test_data = pd.read_csv(os.path.join(CONFIG_PATH_INPUT, 'test.csv'))

    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()

    data_train['tweet'] = data_train.apply(
        lambda x: row_preprocess(x, wordnet_lemmatizer, porter_stemmer), 1)
    data_val['tweet'] = data_val.apply(lambda x: row_preprocess(
        x, wordnet_lemmatizer, porter_stemmer), 1)
    test_data['tweet'] = test_data.apply(
        lambda x: row_preprocess(x, wordnet_lemmatizer, porter_stemmer), 1)

    data_train['label_encoded'] = data_train.apply(lambda x: map_label(x), 1)
    data_val['label_encoded'] = data_val.apply(lambda x: map_label(x), 1)

    train_sentences = data_train.tweet.values
    val_sentences = data_val.tweet.values
    test_sentences = test_data.tweet.values

    train_labels = data_train.label_encoded.values
    val_labels = data_val.label_encoded.values

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)

    train_token_ids, train_attention_masks = torch.tensor(
        getAttentionMask(train_sentences, tokenizer))
    val_token_ids, val_attention_masks = torch.tensor(
        getAttentionMask(val_sentences, tokenizer))
    test_token_ids, test_attention_masks = torch.tensor(
        getAttentionMask(test_sentences, tokenizer))

    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2).to(CONFIG_DEVICE)

    # train_dataloader, validation_dataloader, test_dataloader = mydata_loader(data_train, data_val, data_test, tokenizer)

    train_data = TensorDataset(
        train_token_ids, train_attention_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=CONFIG_BATCH_SIZE)

    validation_data = TensorDataset(
        val_token_ids, val_attention_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=CONFIG_BATCH_SIZE)

    test_data = TensorDataset(test_token_ids, test_attention_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=CONFIG_BATCH_SIZE)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG_LR)

    training(model, CONFIG_EPOCH, train_dataloader,
             validation_dataloader, optimizer)
