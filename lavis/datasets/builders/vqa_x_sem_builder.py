from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.vqa_x_datasets import VQA_X_DATASETS, VQA_X_VAL_DATASETS

@registry.register_builder("vqa_x_sem")
class VQA_X_SEM_Builder(BaseDatasetBuilder):
    train_dataset_cls = VQA_X_DATASETS
    eval_dataset_cls = VQA_X_VAL_DATASETS

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa_x_sem/vqa_x_sem.yaml",
        "eval": "configs/datasets/vqa_x_sem/vqa_x_sem_eval.yaml",
    }