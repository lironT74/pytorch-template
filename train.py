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

EPOCH_PRINT = 300


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


def train(model: nn.Module,
          train_loader: DataLoader,
          eval_loader: DataLoader,
          num_zero_scores_questions: int,
          train_params: TrainParams,
          logger: TrainLogger) -> Metrics:

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
    best_train_loss = 10000

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)
    criterion = nn.NLLLoss()

    print(f"no answers examples in eval {num_zero_scores_questions}")
    print(f"all other examples in eval {len(eval_loader.dataset)}")

    for epoch in tqdm(range(train_params.num_epochs)):

        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        optimizer.zero_grad()

        for i, (image_tensor, question_words_indexes, pad_mask, labels, scores) in enumerate(train_loader):

            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                question_words_indexes = question_words_indexes.cuda()
                labels = labels.cuda()
                scores = scores.cuda()
                pad_mask = pad_mask.cuda()


            if (i+1) % EPOCH_PRINT == 0 or i == 0 or i == len(train_loader) - 1:
                print(f"Epoch: {epoch + 1}, batch: {i+1}/{len(train_loader)} ({cur_time()})")

            image_tensor = image_tensor.squeeze(1)
            y_hat = model((image_tensor, question_words_indexes, pad_mask))

            y_hat_index = torch.argmax(y_hat, dim=1)

            y_multiple_choice_answers_indexes = torch.argmax(scores, dim=1)
            y_multiple_choice_answers = labels[range(labels.shape[0]), y_multiple_choice_answers_indexes]

            loss = criterion(y_hat, y_multiple_choice_answers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            metrics['train_loss'] += loss.item() * labels.size(0)


            occurrences = (y_hat_index.unsqueeze(-1).expand_as(labels) == labels).sum(dim=1)

            for occur in occurrences:
                metrics['train_score'] += get_score(occur.item())


        scheduler.step()

        # # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        with torch.no_grad():
            metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, criterion, num_zero_scores_questions)
        model.train(True)


        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'], metrics['eval_loss'])


        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss'],
                   }


        logger.report_scalars(scalars, epoch, separated=False)


        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)



    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, evaluation_dataloader, criterion, num_zero_scores_questions) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0

    for i, (image_tensor, question_words_indexes, pad_mask, labels, scores) in enumerate(evaluation_dataloader):

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            question_words_indexes = question_words_indexes.cuda()
            labels = labels.cuda()
            scores = scores.cuda()
            pad_mask = pad_mask.cuda()

        image_tensor = image_tensor.squeeze(1)
        y_hat = model((image_tensor, question_words_indexes, pad_mask))

        y_hat_index = torch.argmax(y_hat, dim=1)

        y_multiple_choice_answers_indexes = torch.argmax(scores, dim=1)
        y_multiple_choice_answers = labels[range(labels.shape[0]), y_multiple_choice_answers_indexes]

        loss += criterion(y_hat, y_multiple_choice_answers) * labels.size(0)

        occurrences = (y_hat_index.unsqueeze(-1).expand_as(labels) == labels).sum(dim=1)

        for occur in occurrences:
            score += get_score(occur.item())

        if (i + 1) % EPOCH_PRINT == 0 or i == 0 or i == len(evaluation_dataloader) - 1:
            print(f"---> Evaluation, batch: {i+1}/{len(evaluation_dataloader)} ({cur_time()})")


    score = score / (len(evaluation_dataloader.dataset) + num_zero_scores_questions)

    loss = loss / (len(evaluation_dataloader.dataset))

    return score, loss


