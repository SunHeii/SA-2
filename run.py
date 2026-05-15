# Script to run the models with specified configuration.

import pandas as pd
import hydra
from omegaconf import DictConfig
import os
import json
import time

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC
from models.CASC.main import CASC


def store_run_results(results, path, model_name, ablation):
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

    # Create a json object where we store a classification report as well as raw predictions
    json_object = json.dumps({
        'polarity': results[0],
        'aspect': results[1],
        'predicted_pol': results[2],
        'predicted_asp': results[3],
    })
    with open(f"{path}/{model_name}/run_{date}_{ablation}.json", "w") as outfile:
        outfile.write(json_object)

    # Start the CSV results writing process
    df_output_path = f"{path}/{model_name}/results.csv"

    d = {
      'timestamp': [date, date],
      'type': ['polarity', 'aspect'],
      'ablation': [ablation, ablation],
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


# default: domain=restaurant3 model=SBASC ablation=None]
@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Run SBASC/CASC/WBASC
    if cfg.model.name == 'CASC':
        results = CASC(cfg)(load=cfg.load)
    elif cfg.model.name == 'WBASC':
        results = WBASC(cfg)(load=cfg.load)
    else:
        results=SBASC(cfg)(load=cfg.load, evaluate=False)
    '''else:
        results = SBASC(cfg)(load=cfg.load)
    store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)'''


    # 旁路：注释掉报表保存功能，维持系统静默退出
    # store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)
    print("纯净训练管道执行完毕，未触发任何测试集评估。")


if __name__ == "__main__":
    my_app()
