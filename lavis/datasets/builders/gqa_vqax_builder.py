from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.gqa_sem_datasets import GQA_SEM_DATASETS

from lavis.datasets.datasets.vqa_x_datasets import VQA_X_VAL_DATASETS

@registry.register_builder("gqa_vqax")
class GQA_VQAX_Builder(BaseDatasetBuilder):
    train_dataset_cls = GQA_SEM_DATASETS
    eval_dataset_cls = VQA_X_VAL_DATASETS

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa_sem/gqa_vqax.yaml",
    }