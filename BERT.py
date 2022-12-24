import sys
sys.path.append('.')
import torch

from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import preprocessor as p

from transformers import XLMModel, BertTokenizer, BertForSequenceClassification
from transformers import AdamW

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from config import *
from utils import *
import os

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
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
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
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]

            logits = logits.detach().to(CONFIG_DEVICE).numpy()
            label_ids = b_labels.to(CONFIG_DEVICE).numpy()

            tmp_eval_accuracy = get_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    Validation_Accuracy = (eval_accuracy/nb_eval_steps)
    if(Validation_Accuracy >= best_val_accuracy):
        torch.save(model.state_dict(), CONFIG_SAVE_CHECKPOINTS+'models/BERT_best_model.pt')
        best_val_accuracy = Validation_Accuracy
        print('Model Saved')



if __name__ == '__main__':
    device = CONFIG_DEVICE
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    data_train = pd.read_csv(os.path.join(CONFIG_PATH_INPUT, 'train.csv'))
    data_val = pd.read_csv(os.path.join(CONFIG_PATH_INPUT, 'val.csv'))
    data_test = pd.read_csv(os.path.join(CONFIG_PATH_INPUT, 'test.csv'))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(CONFIG_DEVICE)

    train_dataloader, validation_dataloader, test_dataloader = mydata_loader(data_train, data_val, data_test, tokenizer)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG_LR)
    
    training(model, CONFIG_EPOCH, train_dataloader, validation_dataloader, optimizer)