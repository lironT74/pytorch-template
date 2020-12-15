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

    for epoch in tqdm(range(train_params.num_epochs)):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            image_tensor, q_words_indexes_tensor = x
            label_counts, labels, scores = y

            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                q_words_indexes_tensor = q_words_indexes_tensor.cuda()
                labels = labels.cuda()
                scores = scores.cuda()

            y_hat = model(image_tensor, q_words_indexes_tensor)

            y_multiple_choice_answers_indexes = torch.argmax(scores, dim=1)
            y_multiple_choice_answers = labels[range(labels.shape[0]), y_multiple_choice_answers_indexes]

            loss = nn.NLLLoss(y_hat, y_multiple_choice_answers)
            loss.backward()

            # Optimization step
            if i % batch_size == 0 or i + 1 == len(train_loader):
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
                metrics['train_score'] += get_score(label_counts[y_hat_index])

            metrics['train_loss'] += loss.item()

            # # Report model to tensorboard
            # if epoch == 0 and i == 0:
            #     logger.report_graph(model, x)

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader)

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        # metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader)
        metrics['eval_score'], metrics['eval_loss'] = 0, 0
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)

        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0

    for i, (x, y) in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_hat = model(x)

        loss += nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        score += train_utils.compute_score_with_logits(y_hat, y).sum().item()

    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100

    return score, loss
