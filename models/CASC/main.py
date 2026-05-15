# Main file to run CASC model.
#
# Adapted from Kumar et. al. (2021). Changes have been made to adapt the methods to the
# proposed framework
# https://github.com/Raghu150999/UnsupervisedABSA
#
# Kumar, A., Gupta, P., Balan, R. et al. BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment
# Classification. Neural Process Lett 53, 4207–4224 (2021). https://doi-org.eur.idm.oclc.org/10.1007/s11063-021-10596-6

from hydra import compose, initialize_config_module

from models.CASC.vocab_generator import VocabGenerator
from models.CASC.extracter import Extracter
from models.CASC.score_computer import ScoreComputer
from models.CASC.labeler import Labeler
from models.CASC.trainer import Trainer


class CASC:
    def __init__(self, cfg):
        """
        Creates an instance of the CASC model
        :param cfg: configuration file
        """
        self.name = 'CASC'
        self.cfg = cfg

        self.vocabGenerator = VocabGenerator(cfg)
        self.extracter = Extracter(cfg)
        self.labeler = Labeler(cfg)
        self.trainer = Trainer(cfg, cfg.model.learning_rate, cfg.model.beta1, cfg.model.beta2, cfg.model.batch_size)

    def __call__(self, load=True, evaluate=True):
        """
        Runs the CASC model
        :param load: load in a pre-loaded file for the embeddings. If set to false sentences are passed through the
        encoder
        :param evaluate: evaluate the trained model on the test set.
        :return: evaluaoitn results of trained model
        """
        if not load:
            aspect_vocabularies, sentiment_vocabularies = self.vocabGenerator()
            sentences, aspects, opinions = self.extracter()

            if self.cfg.ablation.name == 'WithoutDeepLearning':
                ScoreComputer(self.cfg, aspect_vocabularies, sentiment_vocabularies)(sentences, aspects, opinions, evaluate=True)
                results = self.labeler(evaluate=True)
            else:
                ScoreComputer(self.cfg, aspect_vocabularies, sentiment_vocabularies)(sentences, aspects, opinions, evaluate=False)
                self.labeler()
                dataset = self.trainer.load_training_data()
                self.trainer.train_model(dataset, epochs=self.cfg.epochs)
                results = self.trainer.evaluate()
        else:
            if self.cfg.ablation.name == 'WithoutDeepLearning':
                results = self.labeler(evaluate=True)
            else:
                dataset = self.trainer.load_training_data()
                self.trainer.train_model(dataset, epochs=self.cfg.epochs)
                results = self.trainer.evaluate()

        return results
      
    def save(self, name):
        """
        Saves the currently trained model
        :param name: filename to store the model
        """
        self.trainer.save_model(name)

    def load(self, name):
        """
        Loads a trained model
        :param name: filename from where to load the model
        """
        self.trainer.load_model(name)

    def hypertuning(self, params):
        """
        Run a hyper tuning instance with a specific hyperparameter configuration
        :param params: the hyperparameter set that will be evaluated
        :return: in-sample loss and accuracy scores of the trained model
        """
        trainer = Trainer(self.cfg, *params)
        dataset = trainer.load_training_data()
        loss, acc = trainer.train_model(dataset, hyper=True, epochs=self.cfg.epochs)

        return loss, acc
      
    def predict(self, sentences):
        """
        Run a predction for unseen sentences on the trained model
        :param sentences: List containing sentences for which we want to make predictions
        :return: predicted sentiment and polarity class for each sentence
        """
        return self.trainer.predict(sentences)


if __name__ == '__main__':
    initialize_config_module(config_module="conf")
    config = compose("config.yaml", overrides=['model=CASC'])
    results = CASC(config)(load=False)
