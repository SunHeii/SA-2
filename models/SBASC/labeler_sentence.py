# sentence labeler for the SBASC model.

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, BertModel
from tqdm import trange
import os


def load_training_data(file_path):
    """
    Load training sentences from a text file
    :param file_path: path to text file with sentences
    :return: list of sentences
    """
    sentences = []
    for line in open(file_path, encoding="utf-8"):
        # Split on end of sentence token to get single sentence on each line and lowercase
        split_lines = list(filter(None, re.split('; |\. |\! |\n| \?', line.lower())))
        for split_line in split_lines:
            sentences.append(split_line.strip())
    return sentences


def load_evaluate_data(path):
    """
    loads in a set of test sentences from a tab delimited file
    :param path: path where the test.txt file is located
    :return: lists with test sentences, and corresponding aspect category and sentiment
    """
    test_sentences = []
    test_cats = []
    test_pols = []

    with open(f'{path}/test.txt', 'r', encoding="utf8") as f:
        for line in f:
            _, cat, pol, sentence = line.strip().split('\t')
            cat = int(cat)
            pol = int(pol)
            test_cats.append(cat)
            test_pols.append(pol)
            test_sentences.append(sentence)
    return test_sentences, test_cats, test_pols


class Labeler:
    def __init__(self, cfg):
        """
        Create a labeler instance used for creating an annotated train set from unstructured data
        :param cfg: configuration object
        """
        self.cfg = cfg
        self.domain = cfg.domain.name
        self.model = SentenceTransformer(cfg.domain.sbert_mapper, device=cfg.device)
        self.cat_threshold = cfg.domain.cat_threshold
        self.pol_threshold = cfg.domain.pol_threshold
        self.root_path = cfg.path_mapper
        self.categories = cfg.domain.aspect_category_mapper
        self.polarities = cfg.domain.sentiment_category_mapper
        self.category_sentences = cfg.domain.aspect_seed_mapper
        self.sentiment_sentences = cfg.domain.sentiment_seed_mapper
        self.labels = None
        self.sentences = None

    def __call__(self, evaluate=True, load=False):
        """
        Perform the actual sentence labeling algorithm
        :param evaluate: whether to evaluate on the test set
        :param load: load in a pre-loaded file for the embeddings. If set to false sentences are passed through the
        encoder
        :return: classificaiton report and predictions for the test set from the labeling algorithm
        """
        category_seeds = self.category_sentences
        polarity_seeds = self.sentiment_sentences

        split = [len(self.categories), len(self.polarities)]

        # Seeds
        seeds = {}
        seeds_len = []
        for cat in self.categories:
            seeds[cat] = list(category_seeds[cat])
            seeds_len.append(len(category_seeds[cat]) + 1)
        for pol in self.polarities:
            seeds[pol] = list(polarity_seeds[pol])
            seeds_len.append(len(polarity_seeds[pol]) + 1)

        # Load and encode the seeds and train set
        self.sentences = load_training_data(f'{self.root_path}/train.txt')

        # Load different embeddings for ablation study
        if self.cfg.ablation.name == 'WithoutSBERT':
            seed_embeddings, embeddings = self.__bert_embedder(load, seeds)
        else:
            seed_embeddings, embeddings = self.__sbert_embedder(load, seeds)

        # Start the similarity finding process
        cosine_scores = []
        for seed_embedding in seed_embeddings:
            # Compute cosine-similarities
            total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
            cosine_scores.append(torch.max(util.cos_sim(total_tensor, embeddings), dim=0)[0].unsqueeze(dim=-1))

        cosine_category_scores, cosine_polarity_scores = torch.split(torch.cat(cosine_scores, 1), split, -1)

        # Get max and argmax of the computed cosine scores
        category_max, category_argmax = cosine_category_scores.max(dim=-1)
        polarity_max, polarity_argmax = cosine_polarity_scores.max(dim=-1)

        labels = np.array(
            [category_argmax.tolist(), category_max.tolist(), polarity_argmax.tolist(), polarity_max.tolist(),
             np.arange(0, len(self.sentences))])

        self.labels = labels

        # df = pd.DataFrame(labels.transpose())
        # df = df.groupby([0, 2]).agg(percentile_cat = (1,lambda x: x.quantile(0.9)), percentile_pol = (3,lambda x: x.quantile(0.9))).reset_index().to_numpy()
        #
        # def select_percentiles(x):
        #     return df[np.where((df[:, 0]==x[0]) & (df[:, 1]==x[2])), 2]
        #
        # def select_percentiles_pol(x):
        #     return df[np.where((df[:, 0]==x[0]) & (df[:, 1]==x[2])), 3]
        #
        # labels = np.r_[labels, np.apply_along_axis(select_percentiles, axis=0, arr=labels).squeeze((0)), np.apply_along_axis(select_percentiles_pol, axis=0, arr=labels).squeeze((0))]
        #

        # Select annotated sentences that are above cossim threshold value
        if self.domain in ['restaurant-nl', "supermarket"]:
            labels = np.transpose(labels[:,
                                  (((labels[2, :] == 1) & (labels[1, :] >= self.cat_threshold)) | (
                                              (labels[2, :] == 0) & (labels[1, :] >= (self.cat_threshold - 0.1)))) &
                                  (((labels[2, :] == 1) & (labels[3, :] >= self.pol_threshold)) | (
                                              (labels[2, :] == 0) & (labels[3, :] >= (self.pol_threshold - 0.1))))])
        else:
            labels = np.transpose(
                labels[:, (labels[1, :] >= self.cat_threshold) & (labels[3, :] >= self.pol_threshold)])

        nf = open(f'{self.root_path}/label-sentences.txt', 'w', encoding="utf8")
        cnt = {}

        # Write labels for annotated sentences to file
        for label in labels:
            sentence = self.sentences[int(label[4])]
            aspect = self.categories[int(label[0])]
            sentiment = self.polarities[int(label[2])]
            nf.write(f'{sentence}\n')
            nf.write(f'{aspect} {sentiment}\n')
            keyword = f'{aspect}-{sentiment}'
            cnt[keyword] = cnt.get(keyword, 0) + 1

        nf.close
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)

        # Start the evaluation process
        if evaluate:
            # Load and encode test sentences
            test_sentences, test_cats, test_pols = load_evaluate_data(self.root_path)
            test_embeddings = self.model.encode(test_sentences, convert_to_tensor=True, show_progress_bar=True)

            cosine_test_scores = []
            for seed_embedding in seed_embeddings:
                # Compute cosine-similarities
                total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
                cosine_test_scores.append(
                    torch.max(util.cos_sim(total_tensor, test_embeddings), dim=0)[0].unsqueeze(dim=-1))

            cosine_category_test_scores, cosine_polarity_test_scores = torch.split(torch.cat(cosine_test_scores, 1),
                                                                                   split, -1)

            category_test_argmax = cosine_category_test_scores.argmax(dim=-1).tolist()
            polarity_test_argmax = cosine_polarity_test_scores.argmax(dim=-1).tolist()

            print("Polarity")
            print(classification_report(
                test_pols, polarity_test_argmax, target_names=self.polarities, digits=4
            ))
            print()

            print("Aspect")
            print(classification_report(
                test_cats, category_test_argmax, target_names=self.categories, digits=4
            ))
            print()

            return classification_report(test_pols, polarity_test_argmax, target_names=self.polarities,
                                         output_dict=True), classification_report(
                test_cats, category_test_argmax, target_names=self.categories, digits=6,
                output_dict=True), polarity_test_argmax, category_test_argmax

    def __bert_embedder(self, load, seeds):
        """
        Use BERT embedding to encode sentences by averaging single word embeddings
        :param load: load embeddings from an already created encoding
        :param seeds: seed sentences to be embedded
        :return: bert embeddings for seeds and all sentences
        """
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.domain.bert_mapper)
        model = BertModel.from_pretrained(self.cfg.domain.bert_mapper, output_hidden_states=True).to(self.cfg.device)

        batch_size = 24  # Set batch size small to prevent out of memory error

        def embedder(sentence_list, device):
            """
            Algorithm to embed a list of sentences using BERT embeddings
            :param sentence_list: list containing sentences to be embedded
            :param device: device to be used for encoding (either CPU or cuda)
            :return: embedded sentences
            """
            sentence_embeddings = []

            for i in trange(0, len(sentence_list), batch_size):
                encoded_dict = tokenizer(
                    sentence_list[i:i + batch_size],
                    padding=True,
                    return_tensors='pt',
                    max_length=128,
                    return_attention_mask=True,
                    truncation=True)
                model.eval()
                with torch.no_grad():
                    outputs = model(encoded_dict['input_ids'].to(device), encoded_dict['token_type_ids'].to(device),
                                    encoded_dict['attention_mask'].to(device))
                # Take the eleventh layer as representation
                x = outputs[2][11]
                mask = encoded_dict['attention_mask'].to(device)  # (bsz, seq_len)
                se = x * mask.unsqueeze(2)
                den = mask.sum(dim=1).unsqueeze(1)
                # Compute average of words as sentence embedding
                sentence_embeddings.extend(se.sum(dim=1) / den)

            return torch.stack(sentence_embeddings)

        # Embed every seed sentences
        seed_embeddings_bert = [embedder(seed, self.cfg.device) for seed in list(seeds.values())]

        if load and os.path.isfile(f'{self.root_path}/bert_avg_train_embeddings.pickle'):
            print(f'Loading embeddings from {self.root_path}')
            embeddings_bert = torch.load(f'{self.root_path}/bert_avg_train_embeddings.pickle')
        else:
            # Embed every sentence in the train set
            embeddings_bert = embedder(self.sentences, self.cfg.device)
            print(f'Saving embeddings to {self.root_path}')
            torch.save(embeddings_bert, f'{self.root_path}/bert_avg_train_embeddings.pickle')

        return seed_embeddings_bert, embeddings_bert

    def __sbert_embedder(self, load, seeds):
        """
        Use Sentence-BERT embedding to encode sentences
        :param load: load embeddings from an already created encoding
        :param seeds: seed sentences to be embedded
        :return: SBERT embeddings for seeds and all sentences
        """
        seed_embeddings = [self.model.encode(seed, convert_to_tensor=True) for seed in list(seeds.values())]

        if load and os.path.isfile(f'{self.root_path}/sbert_train_embeddings.pickle'):
            print(f'Loading embeddings from {self.root_path}')
            embeddings = torch.load(f'{self.root_path}/sbert_train_embeddings.pickle')
        else:
            embeddings = self.model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)
            print(f'Saving embeddings to {self.root_path}')
            torch.save(embeddings, f'{self.root_path}/sbert_train_embeddings.pickle')
        return seed_embeddings, embeddings
