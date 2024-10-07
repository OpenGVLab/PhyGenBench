import json
import os

modelname = 'kelingall'




print(modelname)



directory = '/PhyGenBench/PhyVideos/'
video_directory = os.path.join(directory,modelname)

with open(f'/PhyGenBench/PhyGenEval/single/prompt_replace_augment_single_question_{modelname}_res.json','r') as f:
    single_clip1 = json.load(f)

with open(f'/PhyGenBench/PhyGenEval/multi/prompt_replace_augment_multi_question_{modelname}_res.json','r') as f:
    multi_GPT4o1 = json.load(f)

with open(f'/PhyGenBench/PhyGenEval/multi/prompt_replace_augment_multi_question_{modelname}_res_llava.json','r') as f:
    multi_llava1 = json.load(f)

with open(f'/PhyGenBench/PhyGenEval/video/prompt_replace_augment_video_question_{modelname}_res_intern.json','r') as f:
    video_intern = json.load(f)

with open(f'/PhyGenBench/PhyGenEval/video/prompt_replace_augment_video_question_{modelname}_res.json','r') as f:
    video_GPT4o = json.load(f)


video_GPT_score = [data_tmp['GPT4o_score'] for data_tmp in video_GPT4o]
video_intern_score = [data_tmp["InternVideo2_score"] for data_tmp in video_intern]

single_clip1_score = []

for data_tmp in single_clip1:
    if len(data_tmp["singleimage_question"]) == 1:
        if data_tmp["singleimage_question"][0]["Retrieval Prompt"] != "Last Frame":
            
            score = data_tmp["singleimage_question"][0][modelname] - data_tmp["singleimage_question"][0]["clip_retrieval"]
        else:
            score = data_tmp["singleimage_question"][0][modelname]
        if score >=0 and score < 0.25:
            single_clip1_score.append(0)
        elif score >= 0.25 and score < 0.5:
            single_clip1_score.append(1)
        elif score >=0.5 and score < 0.75:
            single_clip1_score.append(2)
        else:
            single_clip1_score.append(3)
        
    else:
        # len = 2
        single_clip1_score.append(data_tmp[f'{modelname}_descrete'])





multi_GPT1_score = []
for data_tmp in multi_GPT4o1:
    score = 0
    if "CLIP_Index" not in data_tmp["multiimage_question"].keys():
        if 'no' in data_tmp["multiimage_question"]["GPT4o_1"]["Choice"].strip().lower():
            score += 0
        elif 'yes' in data_tmp["multiimage_question"]["GPT4o_1"]["Choice"].strip().lower():
            
            score += 1
        if 'no' in data_tmp["multiimage_question"]["GPT4o_2"]["Choice"].strip().lower():
            
            score += 0
        elif 'yes' in data_tmp["multiimage_question"]["GPT4o_2"]["Choice"].strip().lower():
            
            score += 1
        if 'no' in data_tmp["multiimage_question"]["GPT4o_3"]["Choice"].strip().lower():
            
            score += 0
        elif 'yes' in data_tmp["multiimage_question"]["GPT4o_3"]["Choice"].strip().lower():
            
            score += 1

        multi_GPT1_score.append(score)
    else:
        if data_tmp["multiimage_question"]["GPT4o_3_Score"] >= 1.5:
            score += 1
        else:
            score += 0
        
        if data_tmp["multiimage_question"]["GPT4o_1_Score"] >= 1.5:
            score += 1
        else:
            score += 0
        
        if data_tmp["multiimage_question"]["GPT4o_2_Score"] >= 1.5:
            score += 1
        else:
            score += 0
        
        multi_GPT1_score.append(score)





multi_llava1_score = []

for data_tmp in multi_llava1:
    retrieval_question = data_tmp["multiimage_question"]
    if "CLIP_Index" not in data_tmp["multiimage_question"].keys():
        
        score_discrete = 0
        if 'no' in data_tmp["multiimage_question"]["LLava_1"].strip().lower():

            data_tmp["multiimage_question"]["LLava_1_Score"] = 0
        elif 'yes' in data_tmp["multiimage_question"]["LLava_1"].strip().lower():
            data_tmp["multiimage_question"]["LLava_1_Score"] = 1
            score_discrete += 1
        
        if 'no' in data_tmp["multiimage_question"]["LLava_2"].strip().lower():
            data_tmp["multiimage_question"]["LLava_2_Score"] = 0
        elif 'yes' in data_tmp["multiimage_question"]["LLava_2"].strip().lower():
            data_tmp["multiimage_question"]["LLava_2_Score"] = 1
            score_discrete += 1

        if 'no' in data_tmp["multiimage_question"]["LLava_3"].strip().lower():
            data_tmp["multiimage_question"]["LLava_3_Score"] = 0
        elif 'yes' in data_tmp["multiimage_question"]["LLava_3"].strip().lower():
            data_tmp["multiimage_question"]["LLava_3_Score"] = 1
            score_discrete += 1
        


        multi_llava1_score.append(score_discrete)
    

    else:
        score_discrete = 0

        if 'no' in data_tmp["multiimage_question"]["LLava_3"].strip().lower():
            data_tmp["multiimage_question"]["LLava_3_Score"] = 0
        elif 'yes' in data_tmp["multiimage_question"]["LLava_3"].strip().lower():
            data_tmp["multiimage_question"]["LLava_3_Score"] = 1
            score_discrete += 1


        start_index, max_index, end_index = list(map(int, retrieval_question['CLIP_Index'].split(',')))
        scores1 = []
        for k in range(start_index, max_index+1):
            score1 = 0
            score_re = retrieval_question[f"retrieval_1_{k}"]
            if 'no' in data_tmp["multiimage_question"][f"LLava_1_{k}"].strip().lower():
                score1 = 0
            elif 'yes' in data_tmp["multiimage_question"][f"LLava_1_{k}"].strip().lower():
                score1 = 1
            
            score_final = score_re + score1
            scores1.append(score_final)
        score1_final = max(scores1)

        if score1_final >= 1.5:
            score_discrete += 1
        else:
            score_discrete += 0

        scores2 = []
        for m in range(max_index, end_index):
            score2 = 0
            score_re2 = retrieval_question[f"retrieval_2_{m}"]
            if 'no' in data_tmp["multiimage_question"][f"LLava_2_{m}"].strip().lower():
                score2 = 0
            elif 'yes' in data_tmp["multiimage_question"][f"LLava_2_{m}"].strip().lower():
                score2 = 1
            
            score_final = score_re2 + score2
            scores2.append(score_final)
        score2_final = max(scores2)

        if score2_final >= 1.5:
            score_discrete += 1
        else:
            score_discrete += 0


        data_tmp["multiimage_question"]["LLava_1_Score"] = score1_final
        data_tmp["multiimage_question"]["LLava_2_Score"] = score2_final


        multi_llava1_score.append(score_discrete)

print(len(single_clip1_score),len(multi_GPT1_score),len(video_GPT_score))




# round
vqascore_gpt_gpt_score = [round((single_clip1_score[i] + multi_GPT1_score[i] + video_GPT_score[i]) / 3) for i in range(len(single_clip1_score))]
vqascore_llava_intern_score = [round((single_clip1_score[i] + multi_llava1_score[i] + video_intern_score[i]) / 3) for i in range(len(single_clip1_score))]
average_score = [(vqascore_gpt_gpt_score[i] + vqascore_llava_intern_score[i]) // 2 for i in range(len(vqascore_gpt_gpt_score))]




print(max(single_clip1_score),max(multi_GPT1_score),max(video_GPT_score))
print(max(single_clip1_score),max(multi_llava1_score),max(video_intern_score))
print(max(vqascore_gpt_gpt_score))
print(max(vqascore_llava_intern_score))
print(max(average_score))
alldata1 = []


with open('/PhyGenBench/PhyGenBench/prompts.json','r') as f:
    alldata = json.load(f)

for i in range(len(alldata)):
    data_tmp = alldata[i]
    video_path = os.path.join(video_directory,f'output_video_{i+1}.mp4')
    data_tmp['single'] = single_clip1_score[i]
    data_tmp['multi_gpt'] = multi_GPT1_score[i]
    data_tmp['multi_llava'] = multi_llava1_score[i]
    data_tmp['video_gpt'] = video_GPT_score[i]
    data_tmp['video_intern'] = video_intern_score[i]
    data_tmp[f'{modelname}_average'] = average_score[i]

    data_tmp[f'{modelname}_closed'] = vqascore_gpt_gpt_score[i]
    data_tmp[f'{modelname}_open'] = vqascore_llava_intern_score[i]


    data_tmp['video'] = video_path
    alldata1.append(data_tmp)


with open(f'/PhyGenBench/result/{modelname}.json','w') as f:
    json.dump(alldata1,f,indent=4)



