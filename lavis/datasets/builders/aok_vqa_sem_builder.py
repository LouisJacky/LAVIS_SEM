from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.aok_vqa_datasets_sem import AOKVQADataset
from lavis.datasets.datasets.vqa_x_datasets import VQA_X_VAL_DATASETS

@registry.register_builder("aok_vqa_sem")
class AOKVQA_SEM_Builder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = VQA_X_VAL_DATASETS

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa_sem/aok_vqa_sem.yaml",
    }