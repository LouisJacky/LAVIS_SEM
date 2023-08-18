import os
from collections import OrderedDict
import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import json
import copy


class VQA_X_DATASETS(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image_id": img_id,
            "image": image,

            "instance_id": ann["instance_id"],

            "question_id": ann["question_id"],
            "question": ann["question"],

            "explanation": ann["explanation"],
            "full_explanation": ann["full_explanation"],

            "answers": ann["answers"],
            "confident_answer": ann["confident_answer"]
        }

    def collater(self, samples):
        question_id_list = []
        question_list = []
        image_id_list = []
        image_list = []

        explanation_list = []
        full_explanation_list = []

        answers_list = []
        confident_answer_list = []

        for sample in samples:

            question_id_list.append(sample["question_id"])
            question_list.append(sample["question"])

            image_id_list.append(sample["image_id"])
            image_list.append(sample["image"])

            explanation_list.append(sample["explanation"][0])
            full_explanation_list.append(sample["full_explanation"])

            answers_list.append(sample["answers"])
            confident_answer_list.append(sample["confident_answer"])

        return {
            "question_id": question_id_list,
            "text_input": question_list,

            "image_id": image_id_list,
            "image": torch.stack(image_list, dim=0),

            "text_output_E": explanation_list,
            "full_explanation_list": full_explanation_list,

            "text_output_A": answers_list,
            "text_output_C_A": confident_answer_list
        }

class VQA_X_VAL_DATASETS(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image_id": img_id,
            "image": image,

            "instance_id": ann["instance_id"],

            "question_id": ann["question_id"],
            "question": ann["question"],

            "explanation": ann["explanation"],
            "answers": ann["answers"],
        }

    def collater(self, samples):
        question_id_list = []
        question_list = []
        image_id_list = []
        image_list = []

        explanation_list = []

        answers_list = []

        for sample in samples:
            question_id_list.append(sample["question_id"])
            question_list.append(sample["question"])

            image_id_list.append(sample["image_id"])
            image_list.append(sample["image"])

            explanation_list.append(sample["explanation"])
            answers_list.append(sample["answers"])

        return {
            "question_id": question_id_list,
            "text_input": question_list,

            "image_id": image_id_list,
            "image": torch.stack(image_list, dim=0),

            "text_output_E": explanation_list,
            "text_output_A": answers_list,
        }