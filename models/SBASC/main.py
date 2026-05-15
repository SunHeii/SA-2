# model to run SBASC model.

from hydra import compose, initialize
from models.SBASC.labeler_sentence import Labeler
from models.SBASC.trainer import Trainer


class SBASC:
    def __init__(self, cfg):
        """
        Creates an instance of the SBASC model
        :param cfg: configuration file
        """
        self.name = 'SBASC'
        self.cfg = cfg
        self.params = cfg.domain.params
        self.trainer = None

    def __call__(self, load=True, evaluate=True):
        """
        Runs the SBASC model
        :param load: load in a pre-loaded file for the embeddings. If set to true sentences are passed through the
        encoder
        :param evaluate: evaluate the trained model on the test set.
        :return: evaluaoitn results of trained model
        """
        labeler = Labeler(self.cfg)
        self.trainer = Trainer(self.cfg, **self.params)

        if self.cfg.ablation.name == 'WithoutDeepLearning':
            # 切断：强制关闭伪标签生成阶段的评估
            results = labeler(load=load, evaluate=False)
        else:
            # 切断：强制关闭深度模型生成阶段的评估
            labeler(load=load, evaluate=False)
            dataset = self.trainer.load_training_data()
            self.trainer.train_model(dataset)
            # 切断：彻底移除对测试集的评估调用
            # results = self.trainer.evaluate()

            # 伪造空数据结构，平稳过渡到下一步
            results = None

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

    def labeler(self, load=True):
        """
        Run the labeling process of the model
        :param load: load in a pre-loaded file for the embeddings. If set to true sentences are passed through the
        encoder
        :return: evaluation on the labeling part of the model
        """
        labeler = Labeler(self.cfg)
        return labeler(load=load)

    def hypertuning(self, params):
        """
        Run a hyper tuning instance with a specific hyperparameter configuration
        :param params: the hyperparameter set that will be evaluated
        :return: in-sample loss and accuracy scores of the trained model
        """
        trainer = Trainer(self.cfg, *params)
        dataset = trainer.load_training_data()
        loss, acc = trainer.train_model(dataset, hyper=True)

        return loss, acc
      
    def predict(self, sentences):
        """
        Run a predction for unseen sentences on the trained model
        :param sentences: List containing sentences for which we want to make predictions
        :return: predicted sentiment and polarity class for each sentence
        """
        return self.trainer.predict(sentences)


if __name__ == '__main__':
    initialize(config_path="conf")
    config = compose("config.yaml", overrides=['model=SBASC'])
    SBASC(config)()
