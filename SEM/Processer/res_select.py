import json

tail_name = "blip2_t5"
project_id = "your_project_id"
path_to_LAVIS_SEM ="/path_to_your/LAVIS_SEM"

input_path = f'{path_to_LAVIS_SEM}/lavis/output/SEM/eval/{project_id}/result/test_vqa_result_rank0.json'
# 读取json文件
with open(input_path, 'r') as f:
    data = json.load(f)

total_count = 0
correct_count = 0
# 保存正确答案的question_id
correct_ids = []
for item in data:
    pre_full_explanation = item["pre_full_explanation"]
    if "the answer is " in pre_full_explanation:
        pre_answer = pre_full_explanation.split("the answer is ")[1].split(" because")[0]
    else:
        pre_answer = pre_full_explanation
    if pre_answer in item['answers_gt']:
        correct_ids.append(int(item['question_id']))
        correct_count += 1
    total_count += 1
print("acc:{}%".format(100*correct_count/total_count))

# 写入新的json文件
with open(f'{path_to_LAVIS_SEM}/SEM/Results/correct_question_ids_{tail_name}.json', 'w') as f:
    json.dump(correct_ids, f)