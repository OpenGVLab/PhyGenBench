import os
import json
import base64
import requests
# import jsonlines
from multiprocessing import Pool
from openai import OpenAI
from functools import partial
import time
import os
import base64
import json
import random
import time

from multiprocessing import Pool

from functools import partial
import os
from PIL import Image
import io
import base64
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image






def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_images(img_paths):
    encoded_images = []
    for img_path in img_paths:
        try:
            base64_image = encode_image(img_path)
            encoded_images.append(base64_image)
        except Exception as e:
            print(f'encode image error for {img_path}: {e}')
            return []
    return encoded_images


# 读取视频并获取总帧数
def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

# 平均采样视频帧，包括首尾
def sample_frames(video_path, num_frames):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(frame_rgb), idx))  # 保存帧图像和索引
    
    video.release()
    return frames

# 计算每帧与指定prompt的匹配分数
def calculate_clip_scores(frames, prompt):
    images = [frame[0] for frame in frames]
    print(len(images),prompt)
    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像-文本相似度分数
    return logits_per_image.squeeze().tolist()

# 保存最相似的帧
def save_most_similar_frame(frames, scores, output_path):
    max_index = np.argmax(scores)
    most_similar_frame = frames[max_index][0]
    most_similar_frame.save(output_path)
    print(f"Most similar frame saved at: {output_path}")


# 获取视频的中间帧
def get_middle_frame(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    middle_frame_index = total_frames // 2  # 计算中间帧的位置
    
    video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)  # 设置到中间帧的位置
    
    success, frame = video.read()
    video.release()
    
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError("Failed to retrieve the middle frame from the video.")

# 保存中间帧
def save_middle_frame(video_path, output_path):
    middle_frame = get_middle_frame(video_path)
    middle_frame.save(output_path)
    print(f"Middle frame saved at: {output_path}")


# 获取视频的首帧
def get_first_frame(video_path):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 设置到第一帧的位置
    
    success, frame = video.read()
    video.release()
    
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError("Failed to retrieve the first frame from the video.")

# 保存首帧
def save_first_frame(video_path, output_path):
    first_frame = get_first_frame(video_path)
    first_frame.save(output_path)
    print(f"First frame saved at: {output_path}")



def get_last_frame(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # 设置到最后一帧的位置
    
    success, frame = video.read()
    video.release()
    
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError("Failed to retrieve the last frame from the video.")

# 保存尾帧
def save_last_frame(video_path, output_path):
    last_frame = get_last_frame(video_path)
    last_frame.save(output_path)
    print(f"Last frame saved at: {output_path}")


# 保存最相似帧及其前后两帧
def save_surrounding_frames(frames, scores, output_path):
    max_index = np.argmax(scores)
    # 确定保存帧的索引范围
    start_index = max(max_index - 1, 0)
    end_index = min(max_index + 2, len(frames))

    # 保存帧
    
    for i in range(start_index, end_index):
        frame = frames[i][0]
        frame_output_path = f"{output_path.split('.jpg')[0]}_frame_{i}.jpg"
        frame.save(frame_output_path)
        print(f"Frame saved at: {frame_output_path}")
    return start_index, max_index, end_index



client = OpenAI(
    base_url='',
    api_key='',
)

pretrix = "Answer me in Format:{'Choice':'Yes or No','Reason':'the reason'} "


with open('/PhyGenBench/PhyGenBench/explicit_prompts.json','r') as f:
    explicit_prompt = json.load(f)
# modelname = 'cogvideo2ball'
modelname = 'kelingall'

with open(f'/PhyGenBench/PhyGenEval/multi/prompt_replace_augment_multi_question1_{modelname}_res1_imageclip.json','r') as f:
    data = json.load(f)

directory = '/PhyGenBench/PhyVideos/'

num_frames = 32  # 需要采样的帧数


data = data
result = []

video_directory = os.path.join(directory,modelname)


for i in range(len(data)):
    try:
        T2V_prompt = data[i]["caption"]
        Physical_law = data[i]["physical_laws"]
        
        video_path = os.path.join(video_directory,f'output_video_{i+1}.mp4')
        retrieval_question = data[i]['multiimage_question']
        retrieval_prompt = retrieval_question["Retrieval Prompt"]
        question_prompt1 = retrieval_question["Description1"]
        question_prompt2 = retrieval_question["Description2"]
       
        question_prompt3 = explicit_prompt[i]["explicit_caption"]
        
        output_first_image_path = os.path.join(os.path.join('/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/PhyBench-Videos/multiimage_clips1',modelname),f'output_video_{i+1}_first.jpg')
        output_middle_image_path = os.path.join(os.path.join('/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/PhyBench-Videos/multiimage_clips1',modelname),f'output_video_{i+1}_middle.jpg')
        output_last_image_path = os.path.join(os.path.join('/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/PhyBench-Videos/multiimage_clips1',modelname),f'output_video_{i+1}_last.jpg')


        total_frames = get_video_frame_count(video_path)
        print(f"Total frames in video: {total_frames}")

        # 平均采样帧
        sampled_frames = sample_frames(video_path, num_frames)

        

        if retrieval_prompt == 'Middle Frame':
            

            image_paths1 = [
                output_first_image_path,output_middle_image_path
            ]
            image_paths2 = [
                output_middle_image_path,output_last_image_path
            ]
            image_paths3 = [
                output_first_image_path, output_middle_image_path, output_last_image_path
            ]
            description1 = question_prompt1
            description2 = question_prompt2
            description3 = question_prompt3

            question1 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description1}' + f'\n{pretrix}'

            question2 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description2}' + f'\n{pretrix}'

            question3 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description3}' + f'\n{pretrix}'


            encoded_images1 = encode_images(image_paths1)

            messages_content1 = [{"type": "text", "text": question1}]
            for encoded_image in encoded_images1:
                messages_content1.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                })

            response1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                    },
                    {
                        "role": "user",
                        "content": messages_content1,
                    }
                ],
                max_tokens=200,
            )

            input_string1 = response1.choices[0].message.content

            input_string1 = input_string1.strip().lstrip("```json").rstrip("```").strip()
        

            try:
                parsed_data1 = json.loads(input_string1)
            except:
                print('parsed_data1 = input_string1')
                parsed_data1 = input_string1
            print(parsed_data1)
            data[i]["multiimage_question"]["GPT4o_1"] = parsed_data1






            encoded_images2 = encode_images(image_paths2)

            messages_content2 = [{"type": "text", "text": question2}]
            for encoded_image in encoded_images2:
                messages_content2.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                })

            response2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                    },
                    {
                        "role": "user",
                        "content": messages_content2,
                    }
                ],
                max_tokens=200,
            )


            input_string2 = response2.choices[0].message.content

            input_string2 = input_string2.strip().lstrip("```json").rstrip("```").strip()

            try:
                parsed_data2 = json.loads(input_string2)
            except:
                print('parsed_data2 = input_string2')
                parsed_data2 = input_string2
            print(parsed_data2)

            data[i]["multiimage_question"]["GPT4o_2"] = parsed_data2

            ## all image

            encoded_images3 = encode_images(image_paths3)

            messages_content3 = [{"type": "text", "text": question3}]
            for encoded_image in encoded_images3:
                messages_content3.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                })

            response3 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                    },
                    {
                        "role": "user",
                        "content": messages_content3,
                    }
                ],
                max_tokens=200,
            )


            input_string3 = response3.choices[0].message.content

            input_string3 = input_string3.strip().lstrip("```json").rstrip("```").strip()

            try:
                parsed_data3 = json.loads(input_string3)
            except:
                print('parsed_data3 = input_string3')
                parsed_data3 = input_string3
            print(parsed_data3)

            data[i]["multiimage_question"]["GPT4o_3"] = parsed_data3
            # print('Middle')


        else:
            # scores = calculate_clip_scores(sampled_frames, retrieval_prompt)
            # 保存最相似的帧
            start_index, max_index, end_index = list(map(int, retrieval_question['CLIP_Index'].split(',')))
            scores_res = []
            cnt1 = 0
            scores1 = []
            scores3 = []
            cnt2 = 0
            scores2 = []

            frame_middle_output_paths = []
            frame_middle_retrieval_scores = []

            description1 = question_prompt1
            description2 = question_prompt2
            description3 = question_prompt3

            question1 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description1}' + f'\n{pretrix}'

            question2 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description2}' + f'\n{pretrix}'

            question3 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description3}' + f'\n{pretrix}'
            
            for k in range(start_index, max_index+1):
                

            #     # first-retrieval image
                frame_output_path = f"{output_middle_image_path.split('.jpg')[0]}_frame_{k}.jpg"

                score_re = retrieval_question[f"retrieval_1_{k}"]

                image_paths1 = [
                    output_first_image_path,frame_output_path
                    ]

                question1 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description1}' + f'\n{pretrix}'
                encoded_images1 = encode_images(image_paths1)

                messages_content1 = [{"type": "text", "text": question1}]
                for encoded_image in encoded_images1:
                    messages_content1.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    })

                response1 = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                        },
                        {
                            "role": "user",
                            "content": messages_content1,
                        }
                    ],
                    max_tokens=200,
                    temperature=0,  # 设置为0以确保输出的一致性
                    top_p=1,        # 保持完整的概率分布
                )

                input_string1 = response1.choices[0].message.content

                input_string1 = input_string1.strip().lstrip("```json").rstrip("```").strip()
                try:
                    parsed_data1 = json.loads(input_string1)
                except:
                    print('parsed_data1 = input_string1')
                    parsed_data1 = input_string1
                print(parsed_data1)

                data[i]["multiimage_question"][f"GPT4o_1_{k}"] = parsed_data1

                try:
                    if 'no' in parsed_data1['Choice'].lower():
                        score1 = 0
                    elif 'yes' in parsed_data1['Choice'].lower():
                        score1 = 1
                except:
                    if 'no' in parsed_data1.lower():
                        score1 = 0
                    elif 'yes' in parsed_data1.lower():
                        score1 = 1
                
                score_final = score_re + score1
                scores1.append(score_final)

                frame_middle_output_paths.append(frame_output_path)
                frame_middle_retrieval_scores.append(score_re)
            
            for m in range(max_index, end_index):
                
                frame_output_path = f"{output_middle_image_path.split('.jpg')[0]}_frame_{m}.jpg"

                score_re2 = retrieval_question[f"retrieval_2_{m}"]


                image_paths2 = [
                    frame_output_path,output_last_image_path
                ]

                question2 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description2}' + f'\n{pretrix}'
                encoded_images2 = encode_images(image_paths2)

                messages_content2 = [{"type": "text", "text": question2}]
                for encoded_image in encoded_images2:
                    messages_content2.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    })

                response2 = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                        },
                        {
                            "role": "user",
                            "content": messages_content2,
                        }
                    ],
                    max_tokens=200,
                )


                input_string2 = response2.choices[0].message.content

                input_string2 = input_string2.strip().lstrip("```json").rstrip("```").strip()
                try:
                    parsed_data2 = json.loads(input_string2)
                except:
                    print('parsed_data2 = input_string2')
                    parsed_data2 = input_string2
                print(parsed_data2)

                data[i]["multiimage_question"][f"GPT4o_2_{m}"] = parsed_data2


                try:
                    if 'no' in parsed_data2['Choice'].lower():
                        score2 = 0
                    elif 'yes' in parsed_data2['Choice'].lower():
                        score2 = 1
                except:
                    if 'no' in parsed_data2.lower():
                        score2 = 0
                    elif 'yes' in parsed_data2.lower():
                        score2 = 1
                
                score_final2 = score_re2 + score2
                scores2.append(score_final2)
                

                frame_middle_output_paths.append(frame_output_path)
                frame_middle_retrieval_scores.append(score_re2)


            #     cnt2 += 1
            

            ## all image
            image_paths3 = [output_first_image_path]
            for tmp in frame_middle_output_paths:
                image_paths3.append(tmp)
            image_paths3.append(output_last_image_path)
            
            score_re3 = max(frame_middle_retrieval_scores)

            question3 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description3}' + f'\n{pretrix}'
            encoded_images3 = encode_images(image_paths3)

            messages_content3 = [{"type": "text", "text": question3}]
            for encoded_image in encoded_images3:
                messages_content3.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                })

            response3 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                    },
                    {
                        "role": "user",
                        "content": messages_content3,
                    }
                ],
                max_tokens=200,
                temperature=0,  # 设置为0以确保输出的一致性
                top_p=1,        # 保持完整的概率分布
            )

            input_string3 = response3.choices[0].message.content

            input_string3 = input_string3.strip().lstrip("```json").rstrip("```").strip()
            try:
                parsed_data3 = json.loads(input_string3)
            except:
                print('parsed_data3 = input_string3')
                parsed_data3 = input_string3
            print(parsed_data3)

            data[i]["multiimage_question"][f"GPT4o_3"] = parsed_data3

            try:
                if 'no' in parsed_data3['Choice'].lower():
                    score3 = 0
                elif 'yes' in parsed_data3['Choice'].lower():
                    score3 = 1
            except:
                if 'no' in parsed_data3.lower():
                    score3 = 0
                elif 'yes' in parsed_data3.lower():
                    score3 = 1
            
            score3_final = score_re3 + score3
            



            score1_final = max(scores1)
            score2_final = max(scores2)
            # score3_final = max(scores3)



            data[i]["multiimage_question"]["GPT4o_1_Score"] = score1_final
            data[i]["multiimage_question"]["GPT4o_2_Score"] = score2_final
            data[i]["multiimage_question"]["GPT4o_3_Score"] = score3_final


            

        
        print('ok ',i)
        result.append(data[i])
    except:
        print('error', i)
        result.append(data[i])
    # break

print(len(result))
with open(f'/PhyGenBench/PhyGenEval/multi/prompt_replace_augment_multi_question_{modelname}_res.json','w') as f:
    json.dump(result,f)



