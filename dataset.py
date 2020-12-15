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
    def __init__(self, is_Train: bool) -> None:
        if is_Train:
            self.q_path = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
            self.ann_path = '/home/student/HW2/data/cache/train_target.pkl'
            self.image_path = '/datashare/train2014/'
            self.target_labels_path = '/home/student/HW2/data/cache/train_target.pkl'
            self.all_q_a_path = '/home/student/HW2/data/data_batches_train.pkl'

        else:
            self.q_path = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
            self.ann_path = '/home/student/HW2/data/cache/val_target.pkl'
            self.image_path = '/datashare/val2014/'
            self.target_labels_path = '/home/student/HW2/data/cache/val_target.pkl'
            self.all_q_a_path = '/home/student/HW2/data/data_batches_val.pkl'


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


        with open(self.all_q_a_path, 'rb') as f:
            self.all_q_a = pickle.load(f)


        self.num_of_words = len(self.words2index)

        self.is_Train = is_Train

        self.num_features = len(self.all_q_a)

        self._get_features()

        # # Create list of entries
        # self.entries = self._get_entries()



    def __getitem__(self, index: int) -> Tuple:

        (image_id, question_words_indexes, (label_counts, labels, scores)) = self.all_q_a


        the_image_path = f'{self.image_path}COCO_train2014_{str(image_id).zfill(12)}.jpg'

        image = Image.open(the_image_path)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.squeeze()
            image_tensor = torch.stack([image_tensor, image_tensor, image_tensor])
            image_tensor = image_tensor.unsqueeze(0)


        return (image_tensor, question_words_indexes), (label_counts, labels, scores)



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


        # Make dictionary of questions by image id
        questions_words_indexes_by_image_id = {}

        for q in questions:
            indexes = []

            for word in preprocess_answer(q['question']).split(' '):

                if word in self.words2index:
                    indexes.append(self.words2index[word])
                else:
                    indexes.append(self.words2index['<UNK>'])


            image_id = q['image_id']
            questions_words_indexes_by_image_id[image_id] = torch.tensor(indexes)


        questions_answers_by_image_id = {}

        for target in self.target_labels:

            label_counts, labels, scores = target['label_counts'], target['labels'], target['scores']
            questions_answers_by_image_id[target['image_id']] = (label_counts, torch.tensor(labels), torch.tensor(scores))


        all_q_and_a = []
        for image_id in questions_words_indexes_by_image_id.keys():

            question_words_indexes = questions_words_indexes_by_image_id[image_id]
            label_counts, labels, scores = questions_answers_by_image_id[target['image_id']]

            all_q_and_a.append((image_id, question_words_indexes, (label_counts, labels, scores)))


        if self.is_Train:
            with open(f'/home/student/HW2/data/data_batches_train.pkl', 'wb') as f:
                pickle.dump(all_q_and_a, f)
        else:
            with open(f'/home/student/HW2/data/data_batches_val.pkl', 'wb') as f:
                pickle.dump(all_q_and_a, f)



        # # features = []
        #
        # images_num = len(listdir(self.image_path))
        # print(f"there are {images_num} images ({cur_time()})")
        #
        # for f in listdir(self.image_path):
        #
        #     if isfile(join(self.image_path, f)):
        #
        #         image_id = int(f.split('_')[-1].split('.')[0])
        #         image = Image.open(f'{self.image_path}/{f}')
        #         image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
        #
        #         if image_tensor.shape[1] == 1:
        #             image_tensor = image_tensor.squeeze()
        #             image_tensor = torch.stack([image_tensor, image_tensor, image_tensor])
        #             image_tensor = image_tensor.unsqueeze(0)
        #
        #         q_words_indexes_tensor = questions_words_indexes_by_image_id[image_id]
        #         label_counts, labels, scores = questions_answers_by_image_id[image_id]
        #
        #
        #         cur_example = {
        #                          'x': (image_tensor, q_words_indexes_tensor),
        #                          'y': (label_counts, labels, scores)
        #         }
        #
        #
        #         self.num_features += 1
        #
        #         if self.is_Train:
        #             with open(f'/home/student/HW2/data/data_batches_train/index_{self.num_features}.pkl', 'wb') as f:
        #                 pickle.dump(cur_example, f)
        #         else:
        #             with open(f'/home/student/HW2/data/data_batches_val/index_{self.num_features}.pkl', 'wb') as f:
        #                 pickle.dump(cur_example, f)
        #
        #
        #         print(f"saved exampe {self.num_features}/{images_num} ({cur_time()})")



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
    dataset_train = MyDataset(is_Train=True)
    dataset_val = MyDataset(is_Train=False)

