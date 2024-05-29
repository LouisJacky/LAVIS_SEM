from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.aok_vqa_datasets_sem import AOKVQADataset,AOKVQAEvalDataset

@registry.register_builder("aok_vqa_sem")
class AOKVQA_SEM_Builder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa_sem/aok_vqa_sem.yaml",
        "eval": "configs/datasets/aokvqa_sem/aok_vqa_sem_eval.yaml",
    }