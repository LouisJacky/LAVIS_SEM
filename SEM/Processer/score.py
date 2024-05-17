import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from utils.data_utils import *
import os

project_id = "your_project_id"
tail_name = project_id+"_your_tail_name"
path_to_LAVIS_SEM ="/path_to_your/LAVIS_SEM"

# 设置文件路径
input_path = f'{path_to_LAVIS_SEM}/lavis/output/SEM/eval/{project_id}/result/test_vqa_result_rank0.json'
output_path_full = f'{path_to_LAVIS_SEM}/SEM/Results/vqaX_full_{tail_name}.json'
output_path_exp = f'{path_to_LAVIS_SEM}/SEM/Results/vqaX_exp_{tail_name}.json'
nle_data_test_path = f'{path_to_LAVIS_SEM}/SEM/Datasets/annotation/vqaX_test.json'
resFileExp = f'{path_to_LAVIS_SEM}/SEM/Results/resFileExp_{tail_name}.json'
save_scores_pathExp = f'{path_to_LAVIS_SEM}/SEM/Results/save_scores_pathExp_{tail_name}.json'
annFileExp = f'{path_to_LAVIS_SEM}/SEM/Datasets/annotation/vqaX_test_annot_exp.json'

# 读取并转换json文件
def read_and_transform_json(input_path, output_path_full, output_path_exp):
    with open(input_path, 'r') as f:
        data = json.load(f)

    new_data_full = []
    new_data_exp = []
    total_count = 0
    correct_count = 0

    for item in data:
        new_item_full = {'image_id': int(item['question_id'])}
        new_item_exp = {'image_id': int(item['question_id'])}

        pre_full_explanation = item["pre_full_explanation"]
        if "the answer is " in pre_full_explanation:
            pre_full_explanation = pre_full_explanation.split("the answer is ")[1]
            pre_answer = pre_full_explanation.split(" because")[0]
            pre_explanation = pre_full_explanation.split(" because")[1].split(".")[0] if " because" in pre_full_explanation else ""

        else:
            pre_answer = pre_full_explanation
            pre_explanation = ""

        new_item_full['caption'] = pre_full_explanation
        new_item_exp['caption'] = pre_explanation

        if "the answer is " in item["pre_full_explanation"] and " because" in item["pre_full_explanation"]:
            new_data_full.append(new_item_full)
            new_data_exp.append(new_item_exp)

        if pre_answer in item['answers_gt']:
            correct_count += 1
        total_count += 1

    with open(output_path_full, 'w') as f:
        json.dump(new_data_full, f)
    with open(output_path_exp, 'w') as f:
        json.dump(new_data_exp, f)

    print("Accuracy: {}%".format(100 * correct_count / total_count))

def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions):
    all_file = json.load(open(nle_data_test_path, 'r'))

    gt_answers = {}
    for key, value in all_file.items():
        gt_answers[int(key)] = proc_ans(value['answers'])

    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()

    # # 这段注释的代码将使用微调后解码器输出的答案来判断预测样本是否预测正确，而我们的设计是使用原解码器输出的答案
    # correct_keys = []
    # for key, value in pred_answers.items():
    #     gt_answer = gt_answers[key]
    #     if value == gt_answer:
    #         correct_keys.append(key)
    # 为了避免重复的预测工作，我们将原解码器预测正确的样本id事先保存，在每次需要计算解释分数时加载。
    with open(f'{path_to_LAVIS_SEM}/SEM/Results/correct_question_ids_blip2_t5.json', 'r') as f:
        correct_keys = json.load(f)

    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)

def unfilter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions):
    all_file = json.load(open(nle_data_test_path, 'r'))

    gt_answers = {}
    for key, value in all_file.items():
        gt_answers[int(key)] = proc_ans(value['answers'])

    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()

    # 不过滤预测样本的情况下，我们不需要加载原解码器预测正确的id，直接使用所有预测样本计算解释分数。
    keys = []
    for key, value in pred_answers.items():
        keys.append(key)

    exp_preds = [item for item in exp_predictions if item['image_id'] in keys]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)


# 执行转换和评分
read_and_transform_json(input_path, output_path_full, output_path_exp)
with open(output_path_full, 'r') as f:
    full_predictions = json.load(f)
with open(output_path_exp, 'r') as f:
    exp_predictions = json.load(f)
# unfilter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predicti