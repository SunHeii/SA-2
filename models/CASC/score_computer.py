# score computer to run CASC model.
#
# Adapted from Kumar et. al. (2021). Changes have been made to adapt the methods to the
# proposed framework
# https://github.com/Raghu150999/UnsupervisedABSA
#
# Kumar, A., Gupta, P., Balan, R. et al. BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment
# Classification. Neural Process Lett 53, 4207–4224 (2021). https://doi-org.eur.idm.oclc.org/10.1007/s11063-021-10596-6

from transformers import AutoTokenizer, BertForMaskedLM
from tqdm import tqdm
import torch
from models.CASC.filter_words import filter_words


class ScoreComputer:
    """
    Computes unnormalised overlap scores for each aspect category and sentiment polarity and saves in "scores.txt" file
    """
    def __init__(self, cfg, aspect_vocabularies, sentiment_vocabularies):
        self.cfg = cfg
        self.domain = cfg.domain.name
        self.bert_type = cfg.domain.bert_mapper
        self.device = cfg.device
        self.mlm_model = BertForMaskedLM.from_pretrained(self.bert_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = cfg.path_mapper
        self.aspect_vocabularies = aspect_vocabularies
        self.sentiment_vocabularies = sentiment_vocabularies
    
    def __call__(self, sentences, aspects, opinions, evaluate=False):
        categories = self.cfg.domain.aspect_category_mapper
        polarities = self.cfg.domain.sentiment_category_mapper
        K = self.cfg.model.K_2

        aspect_sets = self.load_vocabulary(self.aspect_vocabularies, self.cfg.domain.M)
        polarity_sets = self.load_vocabulary(self.sentiment_vocabularies, self.cfg.domain.M)

        file = 'scores' if not evaluate else 'scores-test'

        f = open(f'{self.root_path}/{file}.txt', 'w', encoding='utf8')
        
        for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)):
            aspect_words = set()
            opinion_words = set()
            if aspect != '##':
                aspect_words = set(aspect.split())
            if opinion != '##':
                opinion_words = set(opinion.split())
            ids = self.tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids']
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
            word_predictions = self.mlm_model(ids.to(self.device))[0]
            word_scores, word_ids = torch.topk(word_predictions, K, -1)
            word_ids = word_ids.squeeze(0)
            
            cat_scores = {}
            pol_scores = {}

            cntAspects = 0
            cntOpinions = 0

            for idx, token in enumerate(tokens):
                if token in aspect_words:
                    cntAspects += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for cat in categories:
                            if repl in aspect_sets[cat]:
                                cat_scores[cat] = cat_scores.get(cat, 0) + 1
                                break
                if token in opinion_words:
                    cntOpinions += 1
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for repl in replacements:
                        if repl in filter_words or '##' in repl:
                            continue
                        for pol in polarities:
                            if repl in polarity_sets[pol]:
                                pol_scores[pol] = pol_scores.get(pol, 0) + 1
                                break
            summary = f'{sentence}\n'
            for cat in categories:
                val = cat_scores.get(cat, 0) / max(cntAspects, 1)
                summary = summary + f' {cat}: {val}'
            
            for pol in polarities:
                val = pol_scores.get(pol, 0) / max(cntOpinions, 1)
                summary = summary + f' {pol}: {val}'

            f.write(summary)
            f.write('\n')
        f.close()


    def load_vocabulary(self, source, limit):
        target = {}
        for key in source:
            words = []
            for freq, word in source[key][:limit]:
                words.append(word)
            target[key] = set(words)
        return target
