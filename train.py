"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn

from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger

from data_preprocess import get_score
import matplotlib.pyplot

from datetime import datetime
def cur_time():
    now = datetime.now()
    cur_time = now.strftime("%d/%m/%Y %H:%M:%S")
    return cur_time



def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger, batch_size: int) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)

    criterion = nn.NLLLoss()
    # for i, (x,y) in enumerate(train_loader):
    #     print(f'Liron homo #{i+1} ({cur_time()})')
    #     image_tensor, q_words_indexes_tensor = x
    #     label_counts, labels, scores = y
    #     assert len(labels) > 0, "Labels is empty"
    #     assert len(scores) > 0, 'Scores is empty'

    for epoch in tqdm(range(train_params.num_epochs)):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        optimizer.zero_grad()
        print(f"Epoch: {epoch+1}  ({cur_time()})")
        batch_counter = 1
        for i, (x, y) in enumerate(train_loader):
            # print(f'Epoch: {epoch+1}, Image {i+1}/{len(train_loader)}')
            image_tensor, q_words_indexes_tensor = x
            label_counts, labels, scores = y

            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                q_words_indexes_tensor = q_words_indexes_tensor.cuda()
                labels = labels.cuda()
                scores = scores.cuda()

            y_hat = model((image_tensor, q_words_indexes_tensor))

            y_multiple_choice_answers_indexes = torch.argmax(scores, dim=1)
            y_multiple_choice_answers = labels[range(labels.shape[0]), y_multiple_choice_answers_indexes]

            loss = criterion(y_hat, y_multiple_choice_answers)
            loss.backward()
            if i % batch_size == 0 or i == len(train_loader) - 1:

                print(f'Epoch: {epoch+1}, Batch {batch_counter}/{len(train_loader) // batch_size + 1} ({cur_time()})')
                batch_counter += 1
                optimizer.step()
                optimizer.zero_grad()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            # NOTE! This function compute scores correctly only for one hot encoding representation of the logits
            # batch_score = train_utils.compute_score_with_logits(y_hat, y.data).sum()
            y_hat_index = torch.argmax(y_hat, dim=1).item()
            if y_hat_index not in label_counts:
                metrics['train_score'] += 0
            else:
                metrics['train_score'] += get_score(label_counts[y_hat_index]) / len(train_loader)

            metrics['train_loss'] += loss.item()

            # if i > 98:
            #     break
        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader)

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        # Loss is not relevant, so we put it 0
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, criterion)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'], metrics['eval_loss'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss']}
        # print(scalars)

        logger.report_scalars(scalars, epoch, separated=False)

        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)


    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0
    lost_counter = 0
    for i, (x, y) in enumerate(dataloader):
        # print(f'Validation evaluation: {i+1}/{len(dataloader)}')
        image_tensor, q_words_indexes_tensor = x
        label_counts, labels, scores = y

        if len(scores) == 0:
            score += 0
            lost_counter += 1
            continue

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            q_words_indexes_tensor = q_words_indexes_tensor.cuda()
            labels = labels.cuda()

        y_hat = model((image_tensor, q_words_indexes_tensor))

        y_hat_index = torch.argmax(y_hat, dim=1).item()


        if y_hat_index not in label_counts:
            score += 0
        else:
            y_multiple_choice_answers_indexes = torch.argmax(scores, dim=1)
            y_multiple_choice_answers = labels[range(labels.shape[0]), y_multiple_choice_answers_indexes]
            loss += criterion(y_hat, y_multiple_choice_answers).item()
            score += get_score(label_counts[y_hat_index])

        if i % 8000 == 0 or i == len(dataloader) - 1:
            print(f'Evaluation at example #{i+1} ({cur_time()})')

        # TODO: change it
        # if i > 48:
        #     break
    # print(f'Lost {lost_counter} items')

    # TODO: change it in places
    score /= len(dataloader)
    loss /= len(dataloader) - lost_counter

    #
    # score /= 50
    # loss /= 50 - lost_counter


    # score *= 100

    return score, loss
