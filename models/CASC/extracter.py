# extracter to run CASC model.
#
# Adapted from Kumar et. al. (2021). Changes have been made to adapt the methods to the
# proposed framework
# https://github.com/Raghu150999/UnsupervisedABSA
#
# Kumar, A., Gupta, P., Balan, R. et al. BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment
# Classification. Neural Process Lett 53, 4207–4224 (2021). https://doi-org.eur.idm.oclc.org/10.1007/s11063-021-10596-6

from tqdm import tqdm
import spacy


class Extracter:
    '''
    Extract potential-aspects and potential-opinion words
    '''

    def __init__(self, cfg):
        spacy.prefer_gpu()
        self.cfg = cfg
        self.domain = cfg.domain
        if self.domain == 'kto':
            self.smodel = spacy.load('nl_core_news_sm')
        else:
            self.smodel = spacy.load('en_core_web_sm')

        self.root_path = cfg.path_mapper

    def __call__(self, evaluate=False):
        # Extract potential-aspects and potential-opinions
        sentences = []
        aspects = []
        opinions = []

        file = 'train' if not evaluate else 'test'

        with open(f'{self.root_path}/{file}.txt', encoding="utf8") as f:
            for line in tqdm(f):
                text = line.strip()
                sentences.append(text)
                words = self.smodel(text)
                o = []
                a = []
                for word in words:
                    if word.tag_.startswith('ADJ' if self.domain == 'kto' else 'JJ') \
                            or word.tag_.startswith('BW' if self.domain == 'kto' else 'RR'):
                        # Adjective or Adverb
                        o.append(word.text)
                    if word.tag_.startswith('N' if self.domain == 'kto' else 'NN'):
                        # Noun
                        a.append(word.text)
                opinions.append(' '.join(o) if len(o) > 0 else '##')
                aspects.append(' '.join(a) if len(a) > 0 else '##')

        return sentences, aspects, opinions
