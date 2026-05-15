# trainer to run CASC model.
#
# Adapted from Kumar et. al. (2021). Changes have been made to adapt the methods to the
# proposed framework
# https://github.com/Raghu150999/UnsupervisedABSA
#
# Kumar, A., Gupta, P., Balan, R. et al. BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment
# Classification. Neural Process Lett 53, 4207–4224 (2021). https://doi-org.eur.idm.oclc.org/10.1007/s11063-021-10596-6

from transformers import AutoTokenizer
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from models.CASC.model import BERTLinear
from torch import optim
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import re

class Trainer:

    def __init__(self, cfg, learning_rate, beta1, beta2, batch_size):
        self.cfg = cfg
        self.domain = cfg.domain.name
        self.bert_type = cfg.domain.bert_mapper
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = cfg.path_mapper
        self.batch_size = batch_size
        categories = cfg.domain.aspect_category_mapper
        polarities = cfg.domain.sentiment_category_mapper

        self.model = BERTLinear(self.cfg, self.bert_type, len(
            categories), len(polarities)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta1, beta2))

        aspect_dict = {}
        inv_aspect_dict = {}
        for i, cat in enumerate(categories):
            aspect_dict[i] = cat
            inv_aspect_dict[cat] = i

        polarity_dict = {}
        inv_polarity_dict = {}
        for i, pol in enumerate(polarities):
            polarity_dict[i] = pol
            inv_polarity_dict[pol] = i

        self.aspect_dict = aspect_dict
        self.inv_aspect_dict = inv_aspect_dict
        self.polarity_dict = polarity_dict
        self.inv_polarity_dict = inv_polarity_dict

    def load_training_data(self):
        sentences = []
        cats = []
        pols = []
        with open(f'{self.root_path}/label.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    cat, pol = line.strip().split()
                    cats.append(self.inv_aspect_dict[cat])
                    pols.append(self.inv_polarity_dict[pol])
                else:
                    sentences.append(line.strip())
        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        dataset = TensorDataset(
            labels_cat, labels_pol, encoded_dict['input_ids'], encoded_dict['token_type_ids'], encoded_dict['attention_mask'])
        return dataset

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, epochs):
        
        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - self.cfg.domain.validation_data_size, self.cfg.domain.validation_data_size])
        dataloader = DataLoader(train_data, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)

        model = self.model
        device = self.device

        optimizer = self.optimizer

        for epoch in trange(epochs):
            model.train()
            print_loss = 0
            batch_loss = 0
            cnt = 0
            for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in dataloader:
                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _, _ = model(labels_cat.to(device),
                                   labels_pol.to(device), **encoded_dict)
                loss.backward()
                optimizer.step()
                print_loss += loss.item()
                batch_loss += loss.item()
                cnt += 1
                if cnt % 50 == 0:
                    print('Batch loss:', batch_loss / 50)
                    batch_loss = 0

            print_loss /= cnt
            model.eval()
            with torch.no_grad():
                val_loss = 0
                iters = 0
                for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, _, _ = model(labels_cat.to(
                        device), labels_pol.to(device), **encoded_dict)
                    val_loss += loss.item()
                    iters += 1
                val_loss /= iters
            # Display the epoch training loss and validation loss
            print("epoch : {:4}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(
                epoch + 1, epochs, print_loss, val_loss))

    def save_model(self, name):
        torch.save(self.model, f'{self.root_path}/{name}.pth')

    def load_model(self, name):
        self.model = torch.load(f'{self.root_path}/{name}.pth')

    def predict(self, sentences):
        model = self.model
        model.eval()
        device = self.device

        predicted_aspect = []
        predicted_polarity = []

        logits_cats = []
        logits_pols = []

        with torch.no_grad():
            for input in tqdm(sentences):
                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_cat, logits_pol = model(torch.tensor([0]).to(
                    device), torch.tensor([0]).to(device), **encoded_dict)

                predicted_aspect.append(
                    self.aspect_dict[torch.argmax(logits_cat).item()])
                predicted_polarity.append(
                    self.polarity_dict[torch.argmax(logits_pol).item()])

                logits_cats.append(logits_cat)
                logits_pols.append(logits_pol)

        return predicted_aspect, predicted_polarity, torch.cat(logits_cats, 0), torch.cat(logits_pols, 0)

    def predict_multiple(self, sentences, threshold=2):
        split_sentences = []
        indices = []
        # Split sentence at end of sentence character to predict individually
        for i, line in tqdm(enumerate(sentences), desc='Split sentence'):
            split_lines = list(filter(None, re.split('; |\. |\! |\n| \?', line.lower())))
            for split_line in split_lines:
                split_sentences.append(split_line.strip())
                indices.append(i)

        _, predicted_polarity, logits_cat, logits_pol = self.predict(split_sentences)

        # Sum up separated sentences based on indices
        logits_cat_added = torch.zeros(len(sentences), len(self.aspect_dict)).to(self.device)
        logits_cat_added = logits_cat_added.index_add(0, torch.tensor(indices).to(self.device), logits_cat)

        # Only keep aspects where the total log prob value is above threshold
        predicted_aspects = [[] for _ in range(len(sentences))]
        for i, j in zip(*(logits_cat_added > threshold).nonzero(as_tuple=True)):
            predicted_aspects[i].append(self.aspect_dict[j.item()])

        return predicted_aspects, predicted_polarity, logits_cat, logits_pol

    def evaluate(self):
        test_sentences = []
        test_cats = []
        test_pols = []

        with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
            for line in f:
                _, cat, pol, sentence = line.strip().split('\t')
                cat = int(cat)
                pol = int(pol)
                test_cats.append(cat)
                test_pols.append(pol)
                test_sentences.append(sentence)

        df = pd.DataFrame(columns=(
            ['sentence', 'actual category', 'predicted category', 'actual polarity', 'predicted polarity']))

        model = self.model
        model.eval()
        device = self.device

        actual_aspect = []
        predicted_aspect = []

        actual_polarity = []
        predicted_polarity = []

        iters = 0
        with torch.no_grad():
            for input, cat, pol in tqdm(zip(test_sentences, test_cats, test_pols)):

                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_cat, logits_pol = model(torch.tensor([cat]).to(
                    device), torch.tensor([pol]).to(device), **encoded_dict)

                actual_aspect.append(self.aspect_dict[cat])
                actual_polarity.append(self.polarity_dict[pol])

                predicted_aspect.append(
                    self.aspect_dict[torch.argmax(logits_cat).item()])
                predicted_polarity.append(
                    self.polarity_dict[torch.argmax(logits_pol).item()])
                df.loc[iters] = [input, actual_aspect[-1], predicted_aspect[-1],
                                 actual_polarity[-1], predicted_polarity[-1]]
                iters += 1

        df.to_csv(f'{self.root_path}/predictions.csv')

        predicted_pol = np.array(predicted_polarity)
        actual_pol = np.array(actual_polarity)
        print("Polarity")
        print(classification_report(actual_pol, predicted_pol, digits=4))
        print()

        predicted = np.array(predicted_aspect)
        actual = np.array(actual_aspect)
        print("Aspect")
        print(classification_report(actual, predicted, digits=4))
        print()
        
        return classification_report(actual_pol, predicted_pol, digits=6, output_dict=True), classification_report(
            actual, predicted, digits=6, output_dict=True), predicted_pol.tolist(), predicted.tolist()
