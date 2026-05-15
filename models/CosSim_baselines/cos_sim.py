from sklearn.metrics import classification_report
import gensim
from tqdm import tqdm
import codecs
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import pandas as pd
from hydra import compose, initialize

def load_evaluate_data(path):
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

            # sentence = gensim.parsing.preprocessing.remove_stopwords(sentence)
            # sentence = gensim.utils.simple_preprocess(sentence, deacc=True)
            sentence = sentence.strip().split()
            test_sentences.append(sentence)

    return test_sentences, test_cats, test_pols


def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.index_to_key]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []


class Sentences(object):
    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        for line in tqdm(codecs.open(self.filename, "r", encoding="utf-8"), self.filename):
            # line = gensim.parsing.preprocessing.remove_stopwords(line)
            # line = gensim.utils.simple_preprocess(line, deacc=True)
            yield line.strip().split()


class CosSim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain = cfg.domain.name
        self.root_path = cfg.path_mapper
        self.categories = cfg.domain.aspect_category_mapper
        self.polarities = cfg.domain.sentiment_category_mapper
        self.model = None


    def __call__(self, train=True, evaluate=True):
        category_seeds = cfg.domain.aspect_seed_mapper
        polarity_seeds = cfg.domain.sentiment_seed_mapper

        if train:
            print("Training w2v on dataset", self.root_path)
            train_sentences = Sentences(f'{self.root_path}/train.txt')
            self.model = gensim.models.Word2Vec(train_sentences, vector_size=100, window=5, min_count=5, workers=7, sg=0,
                                           negative=5, max_vocab_size=20000)
            self.model.save(f'{self.root_path}/train.w2v')
            print("Training done.")
        else:
            self.model = gensim.models.Word2Vec.load(f'{self.root_path}/train.w2v')

        cat_embeddings = []
        for cat in self.categories:
            seeds = list(category_seeds[cat])
            mean_vector = get_mean_vector(self.model.wv, seeds)
            cat_embeddings.append(mean_vector)
            print(f"{cat} {[w for w, c in self.model.wv.most_similar(mean_vector)]}")
        cat_embeddings = np.stack(cat_embeddings, axis=0)

        pol_embeddings = []
        for pol in self.polarities:
            seeds = list(polarity_seeds[pol])
            mean_vector = get_mean_vector(self.model.wv, seeds)
            pol_embeddings.append(mean_vector)
            print(f"{pol} {[w for w, c in self.model.wv.most_similar(mean_vector)]}")
        pol_embeddings = np.stack(pol_embeddings, axis=0)

        test_sentences, test_cats, test_pols = load_evaluate_data(self.root_path)

        test_embeddings = []
        for test_sentence in test_sentences:
            embedding = get_mean_vector(self.model.wv, test_sentence)
            if len(embedding) > 0:
                test_embeddings.append(embedding)
            else:
                test_embeddings.append(np.zeros(self.model.layer1_size))

        test_embeddings = np.stack(test_embeddings, axis=0)

        cat_pred = np.argmax(cosine_similarity(cat_embeddings, test_embeddings), axis=0)
        pol_pred = np.argmax(cosine_similarity(pol_embeddings, test_embeddings), axis=0)

        print("Polarity")
        print(classification_report(
            test_pols, pol_pred, target_names=cfg.domain.sentiment_category_mapper, digits=4, zero_division=0
        ))
        print()

        print("Aspect")
        print(classification_report(
            test_cats, cat_pred, target_names=cfg.domain.aspect_category_mapper, digits=4, zero_division=0
        ))
        print()

        return classification_report(
            test_pols, pol_pred, target_names=cfg.domain.sentiment_category_mapper, digits=4, zero_division=0, output_dict=True
        ), classification_report(
            test_cats, cat_pred, target_names=cfg.domain.aspect_category_mapper, digits=4, zero_division=0, output_dict=True
        )


def store_run_results(results, path, model_name, ablation, domain):
    """
    Function that stores the results of a model run in a JSON file and appends key metrics to a table
    :param results: result output from the model
    :param path: path to results directory
    :param model_name: name of the model used
    :param ablation: the ablation used when training the model (will be 'None' if the full model is used)
    """
    date = time.strftime("%Y%m%d_%H%M%S")

    # Create the folder if it does not exist
    if not os.path.exists(f"{path}/{model_name}/"):
        os.makedirs(f"{path}/{model_name}")

    # Start the CSV results writing process
    df_output_path = f"{path}/{model_name}/results.csv"

    d = {
      'timestamp': [date, date],
      'type': ['polarity', 'aspect'],
      'ablation': [ablation, ablation],
      'domain': [domain, domain],
      'accuracy': [results[0]['accuracy'], results[1]['accuracy']],
      'precision': [results[0]['macro avg']['precision'], results[1]['macro avg']['precision']],
      'recall': [results[0]['macro avg']['recall'], results[1]['macro avg']['recall']],
      'f1-score': [results[0]['macro avg']['f1-score'], results[1]['macro avg']['f1-score']]
    }

    # Create the dataframe
    df_new = pd.DataFrame(data=d)
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], format='%Y%m%d_%H%M%S')

    # Create a new results file if it does not exist, otherwise append to the existing file
    if os.path.exists(df_output_path):
      df_old = pd.read_csv(df_output_path, index_col=0)
      pd.concat([df_old, df_new]).reset_index(drop=True).to_csv(df_output_path, mode='w')
    else:
      df_new.to_csv(df_output_path, mode='w')


if __name__ == '__main__':
    with initialize(config_path="conf"):
        cfg = compose("config.yaml",
                      overrides=[f'domain=restaurant3'])
        cos_sim = CosSim(cfg)
        RUNS = 5
        polarity_list, aspect_list = [], []
        for i in range(RUNS):
            polarity, aspect = cos_sim(train=True)
            polarity_list.append(polarity)
            aspect_list.append(aspect)
            store_run_results([polarity, aspect], f'./results', 'cos_sim', 'none', cfg.domain.name)

        acc, prec, rec, f1 = 0, 0, 0, 0
        for item in polarity_list:
            acc += item['accuracy']
            prec += item['macro avg']['precision']
            rec += item['macro avg']['recall']
            f1 += item['macro avg']['f1-score']

        print(f"accuracy: {acc/len(polarity_list)},\t precision: {prec/len(polarity_list)},\t recall: {rec/len(polarity_list)},\t f1-score: {f1/len(polarity_list)}")

        acc, prec, rec, f1 = 0, 0, 0, 0
        for item in aspect_list:
            acc += item['accuracy']
            prec += item['macro avg']['precision']
            rec += item['macro avg']['recall']
            f1 += item['macro avg']['f1-score']

        print(
            f"accuracy: {acc / len(aspect_list)},\t precision: {prec / len(aspect_list)},\t recall: {rec / len(aspect_list)},\t f1-score: {f1 / len(aspect_list)}")
