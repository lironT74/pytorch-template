"""
Here, we create a custom dataset
"""
import torch
import pickle
import json
from lang import Lang

from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
from torchvision import models, transforms
from PIL import Image

from os import listdir
from os.path import isfile, join


from data_preprocess import *

from datetime import datetime
def cur_time():
    now = datetime.now()
    cur_time = now.strftime("%d/%m/%Y %H:%M:%S")
    return cur_time



BATCH_SIZE = 5000

class MyDataset(Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, mode: str, emb_dropout=0.4, is_pre=False, only_lstm=False) -> None:
        self.mode = mode

        if self.mode == "train":
            self.q_path = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
            self.ann_path = '/home/student/HW2/data/cache/train_target.pkl'
            self.image_path = '/datashare/train2014/'
            self.target_labels_path = '/home/student/HW2/data/cache/train_target.pkl'
            self.all_q_a_path = '/home/student/HW2/data/data_batches_train.pkl'
            self.zero_scores_questions_path = f'/home/student/HW2/data/zero_scores_questions_train.pkl'

        else:
            self.q_path = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
            self.ann_path = '/home/student/HW2/data/cache/val_target.pkl'
            self.image_path = '/datashare/val2014/'
            self.target_labels_path = '/home/student/HW2/data/cache/val_target.pkl'
            self.all_q_a_path = '/home/student/HW2/data/data_batches_val.pkl'
            self.zero_scores_questions_path = f'/home/student/HW2/data/zero_scores_questions_val.pkl'



        with open('/home/student/HW2/data/cache/trainval_ans2label.pkl', 'rb') as f:
            self.ans2label = pickle.load(f)

        with open('/home/student/HW2/data/cache/trainval_label2ans.pkl', 'rb') as f:
            self.label2ans = pickle.load(f)

        with open('/home/student/HW2/data/cache/trainval_words2index.pkl', 'rb') as f:
            self.words2index = pickle.load(f)

        with open('/home/student/HW2/data/cache/trainval_index2words.pkl', 'rb') as f:
            self.index2words = pickle.load(f)

        with open(self.target_labels_path, 'rb') as f:
            self.target_labels = pickle.load(f)

        self.emb_dropout = emb_dropout


        if not is_pre:
            with open(self.all_q_a_path, 'rb') as f:
                self.all_q_a = pickle.load(f)

            with open(self.zero_scores_questions_path, 'rb') as f:
                self.zero_scores_questions = pickle.load(f)


        self.only_lstm = only_lstm

        self.num_of_words = len(self.words2index)

        self.num_of_labels = len(self.ans2label)

        self.max_len_q = 20



        if not is_pre:
            self.num_features = len(self.all_q_a)
        else:
            self._get_features()


        # # Create list of entries
        # self.entries = self._get_entries()



    def __getitem__(self, index: int) -> Tuple:
        (image_id, question_words_indexes, pad_mask, labels, scores) = self.all_q_a[index]


        # return question_words_indexes, pad_mask, labels, scores

        # y_multiple_choice_answers_indexes = torch.argmax(scores, dim=1)
        # y_multiple_choice_answers = labels[range(labels.shape[0]), y_multiple_choice_answers_indexes]

        # if self.mode == 'train':
        #
        #     # for i in range(len(question_words_indexes)):
        #     #     if question_words_indexes[i] == self.words2index['<PAD>']:
        #     #         break
        #     #     if np.random.binomial(n=1, p=self.emb_dropout):
        #     #         question_words_indexes[i] = self.words2index['<UNK>']
        #
        #
        #     image_tensor = torch.load(f"/home/student/HW2/data/train_tensors/COCO_train2014_{str(image_id).zfill(12)}_tensor")
        #
        # else:
        #     image_tensor = torch.load(f"/home/student/HW2/data/val_tensors/COCO_val2014_{str(image_id).zfill(12)}_tensor")
        #

        the_image_path = f'{self.image_path}COCO_{self.mode}2014_{str(image_id).zfill(12)}.jpg'

        image = Image.open(the_image_path)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.squeeze()
            image_tensor = torch.stack([image_tensor, image_tensor, image_tensor])
            image_tensor = image_tensor.unsqueeze(0)


        return image_tensor, question_words_indexes, pad_mask, labels, scores



    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return self.num_features


    def _get_features(self) -> Any:
        """
        Load all features into a structure (not necessarily dictionary). Think if you need/can load all the features
        into the memory.
        :return:
        :rtype:
        """

        # Read json of questions
        with open(self.q_path, "r") as q_file:
            questions = json.load(q_file)['questions']


        # Make dictionary of questions by image_path id
        questions_words_indexes_by_image_id = {}

        for q in questions:
            indexes = []
            pad_mask = []

            for word in preprocess_answer(q['question']).split(' '):

                if word in self.words2index:
                    indexes.append(self.words2index[word])
                else:
                    indexes.append(self.words2index['<UNK>'])

                pad_mask.append(False)

            for i in range(len(preprocess_answer(q['question']).split(' ')), self.max_len_q, 1):
                indexes.append(self.words2index['<PAD>'])
                pad_mask.append(True)


            image_id = q['image_id']
            questions_words_indexes_by_image_id[image_id] = (torch.tensor(indexes), torch.tensor(pad_mask))


        questions_answers_by_image_id = {}

        zero_scores_questions = []
        for target in self.target_labels:

            # label_counts, labels, scores = target['label_counts'], target['labels'], target['scores']
            # questions_answers_by_image_id[target['image_id']] = (label_counts, torch.tensor(labels), torch.tensor(scores))

            labels, scores = target['labels'], target['scores']

            labels = [int(i) for i in labels]
            scores = [float(i) for i in scores]

            if len(scores) == 0:
                zero_scores_questions.append(target['image_id'])

            for i in range(len(labels), 10):
                labels.append(-1)
                scores.append(-1.0)

            questions_answers_by_image_id[target['image_id']] = (torch.tensor(labels), torch.tensor(scores))


        all_q_and_a = []
        all_q_a_no_answer = []

        for image_id in questions_words_indexes_by_image_id.keys():

            question_words_indexes, pad_mask = questions_words_indexes_by_image_id[image_id]
            labels, scores = questions_answers_by_image_id[image_id]

            if self.mode == "train":
                if image_id not in zero_scores_questions:
                    all_q_and_a.append((image_id, question_words_indexes, pad_mask, labels, scores))
            else:
                if image_id not in zero_scores_questions:
                    all_q_and_a.append((image_id, question_words_indexes, pad_mask, labels, scores))
                else:
                    all_q_a_no_answer.append((image_id, question_words_indexes, pad_mask, labels, scores))



        print(f"{len(all_q_and_a)} examples")
        print(f"{len(zero_scores_questions)} examples with no answers")

        if self.mode == 'train':

            with open(f'/home/student/HW2/data/data_batches_train.pkl', 'wb') as f:
                pickle.dump(all_q_and_a, f)

            # with open(f'/home/student/HW2/data/zero_scores_questions_train.pkl', 'wb') as f:
            #     pickle.dump(zero_scores_questions, f)


        else:

            with open(f'/home/student/HW2/data/data_batches_val.pkl', 'wb') as f:
                pickle.dump(all_q_and_a, f)

            with open(f'/home/student/HW2/data/data_batches_no_answer_val.pkl', 'wb') as f:
                pickle.dump(all_q_a_no_answer, f)


            # with open(f'/home/student/HW2/data/zero_scores_questions_val.pkl', 'wb') as f:
            #     pickle.dump(zero_scores_questions, f)




    # def _get_entries(self) -> List:
    #     """
    #     This function create a list of all the entries. We will use it later in __getitem__
    #     :return: list of samples
    #     """
    #     entries = []
    #
    #     for idx, item in self.features.items():
    #         entries.append(self._get_entry(item))
    #
    #     return entries
    #
    #
    #
    # @staticmethod
    # def _get_entry(item: Dict) -> Dict:
    #     """
    #     :item: item from the data. In this example, {'input': Tensor, 'y': int}
    #     """
    #     x = item['input']
    #     y = torch.Tensor([1, 0]) if item['label'] else torch.Tensor([0, 1])
    #
    #     return {'x': x, 'y': y}


if __name__ == '__main__':
    dataset_train = MyDataset(mode='train', is_pre=True)
    dataset_val = MyDataset(mode='eval', is_pre=True)

