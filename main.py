"""
Main file
We will run the whole program from here
"""

import torch
import hydra
import os

from train import train
import train_separately
from train_special_loss import train_special_loss
from dataset import MyDataset
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
from VQA_model_first import VQA
from VQA_model_attention import VQA_Attention
from torchvision import models, transforms
from PIL import Image
from multiprocessing import Pool
from VQA_form_lecture import VQA_from_lecture

from LSTM_question_model import LSTM

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

    val_dataset = MyDataset(is_Train=False)
    eval_loader = DataLoader(val_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=cfg['main']['num_workers'])


    train_dataset = MyDataset(is_Train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=cfg['main']['num_workers'])


    word_vocab_size = train_dataset.num_of_words
    num_clases = train_dataset.num_of_labels


    # Init model

    model = VQA_Attention(word_vocab_size=word_vocab_size, num_classes=num_clases)

    # lstm_model = LSTM(word_vocab_size=word_vocab_size, num_classes=num_clases)

    # model_path = '/home/student/HW2/logs/my_exp_12_23_12_53_30/model.pth'
    # pretrained_lstm_dict = torch.load(model_path)['model_state']
    # lstm_model.load_state_dict(pretrained_lstm_dict)
    #
    # model.question_model = lstm_model.lstm_model
    # model.word_embedding = lstm_model.word_embedding

    # for param in model.features.parameters():
    #     param.requires_grad = True


    # TODO: Add gpus_to_use
    # if cfg['main']['parallel']:
    #     model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, eval_loader, train_params, logger, cfg['train']['batch_size'])

    # metrics = train(model, train_loader, eval_loader, train_params, logger, cfg['train']['batch_size'])
    # metrics = train_special_loss(model, train_loader, eval_loader, train_params, logger, cfg['train']['batch_size'])

    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)




def save_images_aux(image_path):

    print(image_path)

    image = Image.open('/datashare/train2014/' + image_path)
    # image_tensor = transforms.ToTensor()(image_path)  # unsqueeze to add artificial first dimension
    image = transforms.Resize((224, 224))(image)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.squeeze()
        image_tensor = torch.stack([image_tensor, image_tensor, image_tensor])
        image_tensor = image_tensor.unsqueeze(0)

    torch.save(image_tensor, "/home/student/HW2/data/train_tensors/" + image_path[:-4] + "_tensor")



@hydra.main(config_path="config", config_name='config')
def main_lstm(cfg: DictConfig) -> None:
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
    val_dataset = MyDataset(is_Train=False, only_lstm=True)
    train_dataset = MyDataset(is_Train=True, only_lstm=True)
    word_vocab_size = train_dataset.num_of_words
    print('word vocab size:', word_vocab_size)

    num_clases = train_dataset.num_of_labels

    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=cfg['main']['num_workers'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=cfg['main']['num_workers'])

    # Init model

    # model = VQA(word_vocab_size=word_vocab_size, num_classes=num_clases)
    model = LSTM(word_vocab_size=word_vocab_size, num_classes=num_clases)


    # TODO: Add gpus_to_use
    # if cfg['main']['parallel']:
    #     model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train_separately.train(model, train_loader, eval_loader, train_params, logger, cfg['train']['batch_size'])

    # metrics = train(model, train_loader, eval_loader, train_params, logger, cfg['train']['batch_size'])
    # metrics = train_special_loss(model, train_loader, eval_loader, train_params, logger, cfg['train']['batch_size'])

    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)






if __name__ == '__main__':
    main()


    # jobs = []
    # for image_path in os.listdir('/datashare/train2014/'):
    #     if os.path.exists("/home/student/HW2/data/train_tensors/" + image_path[:-4] + "_tensor"):
    #         continue
    #     jobs.append(image_path)
    #
    # with Pool() as pool:
    #     print(f"There are {len(jobs)} images.")
    #     pool.map(save_images_aux, jobs)
    #     pool.close()
    #     pool.join()