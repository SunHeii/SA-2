# labeler to run CASC model.
#
# Adapted from Kumar et. al. (2021). Changes have been made to adapt the methods to the
# proposed framework
# https://github.com/Raghu150999/UnsupervisedABSA
#
# Kumar, A., Gupta, P., Balan, R. et al. BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment
# Classification. Neural Process Lett 53, 4207–4224 (2021). https://doi-org.eur.idm.oclc.org/10.1007/s11063-021-10596-6

import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Labeler:

    def __init__(self, cfg):
        self.cfg = cfg
        self.domain = cfg.domain.name
        self.root_path = cfg.path_mapper
    
    def __call__(self, evaluate=False):
        categories = self.cfg.domain.aspect_category_mapper
        polarities = self.cfg.domain.sentiment_category_mapper

        # Distributions
        dist = {}
        for cat in categories:
            dist[cat] = []
        for pol in polarities:
            dist[pol] = []

        # Read scores
        file = 'scores' if not evaluate else 'scores-test'

        with open(f'{self.root_path}/scores.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    values = line.strip().split()
                    for j, val in enumerate(values):
                        if j % 2 == 1:
                            dist[values[j-1][:-1]].append(float(val))
        
        # Compute mean and sigma for each category
        means = {}
        sigma = {}
        for key in dist:
            means[key] = np.mean(dist[key])
            sigma[key] = np.std(dist[key])

        if not evaluate:
            nf = open(f'{self.root_path}/label.txt', 'w', encoding="utf8")
            cnt = {}
            with open(f'{self.root_path}/{file}.txt', 'r', encoding="utf8") as f:
                sentence = None
                for idx, line in enumerate(f):
                    if idx % 2 == 1:
                        aspect = []
                        sentiment = []
                        key = None
                        for j, val in enumerate(line.strip().split()):
                            if j % 2 == 1:
                                # Normalise score
                                dev = (float(val) - means[key]) / sigma[key]
                                if dev >= self.cfg.model.lambda_threshold:
                                    if key in categories:
                                        aspect.append(key)
                                    else:
                                        sentiment.append(key)
                            else:
                                key = val[:-1]
                        # No conflict (avoid multi-class sentences)
                        if len(aspect) == 1 and len(sentiment) == 1:
                            nf.write(sentence)
                            nf.write(f'{aspect[0]} {sentiment[0]}\n')
                            keyword = f'{aspect[0]}-{sentiment[0]}'
                            cnt[keyword] = cnt.get(keyword, 0) + 1
                    else:
                        sentence = line
            nf.close()
            # Labeled data statistics
            print('Labeled data statistics:')
            print(cnt)
        else:
            cnt = {}
            aspects = []
            sentiments = []

            with open(f'{self.root_path}/{file}.txt', 'r', encoding="utf8") as f:
                for idx, line in enumerate(f):
                    if idx % 2 == 1:
                        aspect = []
                        sentiment = []
                        aspect_high = -100
                        sentiment_high = -100
                        key = None
                        for j, val in enumerate(line.strip().split()):
                            if j % 2 == 1:
                                # Normalise score
                                dev = (float(val) - means[key]) / sigma[key]
                                if key in categories and dev > aspect_high:
                                    aspect = [key]
                                    aspect_high = dev
                                elif key in polarities and dev > sentiment_high:
                                    sentiment = [key]
                                    sentiment_high = dev
                            else:
                                key = val[:-1]
                        # No conflict (avoid multi-class sentences)
                        if len(aspect) == 1 and len(sentiment) == 1:
                            keyword = f'{aspect[0]}-{sentiment[0]}'
                            cnt[keyword] = cnt.get(keyword, 0) + 1
                            aspects.append(aspect[0])
                            sentiments.append(sentiment[0])
                            
            # Labeled data statistics
            print('Labeled data statistics:')
            print(cnt)
            
            test_cats = []
            test_pols = []

            with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
                for line in f:
                    _, cat, pol, sentence = line.strip().split('\t')
                    cat = int(cat)
                    pol = int(pol)
                    test_cats.append(list(categories)[cat])
                    test_pols.append(list(polarities)[pol])

            predicted = np.array(sentiments)
            actual = np.array(test_pols)
            print("Polarity")
            print(classification_report(actual, predicted, digits=4))
            print()

            predicted = np.array(aspects)
            actual = np.array(test_cats)
            print("Aspect")
            print(classification_report(actual, predicted, digits=4))
            print()
            
            cm = confusion_matrix(actual, predicted, labels=list(categories))
            cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(categories))

            fig, ax = plt.subplots(figsize=(15, 15))
            cmp.plot(ax=ax, xticks_rotation='vertical')
