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

class MyDataset(Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, is_Train: bool, lang=None) -> None:
        if is_Train:
            self.q_path = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
            self.ann_path = '/datashare/v2_mscoco_train2014_annotations.json'
            self.image_path = '/datashare/train2014'
        else:
            self.q_path = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
            self.ann_path = '/datashare/v2_mscoco_val2014_annotations.json'
            self.image_path = '/datashare/val2014'

        self.is_Train = is_Train

        assert self.is_Train and lang is None or not self.is_Train and lang is not None, "Something is wrong with is train ang lang module"
        self.lang = Lang() if self.is_Train else lang

        # Load features
        self.features = self._get_features()

        # Create list of entries
        # self.entries = self._get_entries()



    def __getitem__(self, index: int) -> Tuple:
        return self.entries[index]['x'], self.entries[index]['y']

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.entries)

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

        with open(self.ann_path, "r") as ann_file:
            annotations = json.load(ann_file)['annotations']


        # Make dictionary of questions by image id
        questions_by_image_id = {}
        all_questions_ids = []

        for q in questions:
            question_str = q['question']
            if self.is_Train:
                self.lang.add_sentence(question_str)

            all_questions_ids.append(q["question_id"])

        for q in questions:
            image_id = q['image_id']
            q_embedding = self.lang.sentence_embedding(q['question'])
            questions_by_image_id[image_id] = (q['question_id'], q_embedding)


        all_possible_answers = list()
        print(len(annotations))
        for annot in annotations:
            if annot["question_id"] in all_questions_ids:
                all_possible_answers.append(annot["multiple_choice_answer"])
            else:
                print(f"oh no")
                # for answer in annot["answers"]:
                #     print(answer)
                #     all_possible_answers.add(answer["answer"])

        print(len(set(all_possible_answers)))


        features = []

        for f in listdir(self.image_path):

            if isfile(join(self.image_path, f)):

                image_id = int(f.split('_')[-1].split('.')[0])
                image = Image.open(f'{self.image_path}/{f}')
                image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension

                if image.shape[1] == 1:
                    image = image.squeeze()
                    image_tensor = torch.stack([image, image, image])
                    image_tensor = image_tensor.unsqueeze(0)

                features.append({
                                 'x': ((image_id, image_tensor), questions_by_image_id[image_id]),
                                 'y': ()
                })

        return features

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        entries = []

        for idx, item in self.features.items():
            entries.append(self._get_entry(item))

        return entries

    @staticmethod
    def _get_entry(item: Dict) -> Dict:
        """
        :item: item from the data. In this example, {'input': Tensor, 'y': int}
        """
        x = item['input']
        y = torch.Tensor([1, 0]) if item['label'] else torch.Tensor([0, 1])

        return {'x': x, 'y': y}

if __name__ == '__main__':
    dataset = MyDataset(is_Train=True)

