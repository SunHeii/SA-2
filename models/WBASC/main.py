from hydra import compose, initialize

from models.WBASC.labeler import Labeler
from models.SBASC.trainer import Trainer


class WBASC:
    def __init__(self, cfg):
        """
        Creates an instance of the WBASC model
        :param cfg: configuration file
        """
        self.name = 'c'
        self.cfg = cfg
        self.params = cfg.domain.params

    def __call__(self, load=True, evaluate=True):
        """
        Runs the WBASC model
        :param load: load in a pre-loaded file for the embeddings. If set to true sentences are passed through the
        encoder
        :param evaluate: evaluate the trained model on the test set.
        :return: evaluaoitn results of trained model
        """
        labeler = Labeler(self.cfg)
        trainer = Trainer(self.cfg, **self.params)

        if self.cfg.ablation.name == 'WithoutDeepLearning':
            results = labeler(load=load, evaluate=True)
        else:
            labeler(load=load, evaluate=evaluate)
            dataset = trainer.load_training_data(file='label-sbert.txt')
            trainer.train_model(dataset)
            results = trainer.evaluate()

        return results
    
    def labeler(self, load=True):
        """
        Run the labeling process of the model
        :param load: load in a pre-loaded file for the embeddings. If set to true sentences are passed through the
        encoder
        :return: evaluation on the labeling part of the model
        """
        labeler = Labeler(self.cfg)
        labeler(load=load)

    def hypertuning(self, params):
        """
        Run a hyper tuning instance with a specific hyperparameter configuration
        :param params: the hyperparameter set that will be evaluated
        :return: in-sample loss and accuracy scores of the trained model
        """
        trainer = Trainer(self.cfg, *params)
        dataset = trainer.load_training_data(file='label-sbert.txt')
        loss, acc = trainer.train_model(dataset, hyper=True)

        return loss, acc


if __name__ == '__main__':
    initialize(config_path="conf")
    config = compose("config.yaml", overrides=['model=WBASC'])
    WBASC(config)()
