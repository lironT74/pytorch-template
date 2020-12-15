"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from train import train
from dataset import MyDataset
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
from VQA_model_liron import VQA


torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load dataset
    # train_dataset = MyDataset(path=cfg['main']['paths']['train'])
    train_dataset = MyDataset(is_Train=True)
    word_vocab_size = train_dataset.num_of_words
    num_clases = train_dataset.num_of_labels
    print(train_dataset.label2ans[3])

    # val_dataset = MyDataset(path=cfg['main']['paths']['validation'])
    train_loader = DataLoader(train_dataset, 1, shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    # eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
    #                          num_workers=cfg['main']['num_workers'])

    # Init model
    model = VQA(word_vocab_size=word_vocab_size, num_classes=num_clases)

    # TODO: Add gpus_to_use
    # if cfg['main']['parallel']:
    #     model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, None, train_params, logger, cfg['train']['batch_size'])
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
