import argparse
import torch
import os
import json
from tqdm import tqdm
import copy
# import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math
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

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image
# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("/mnt/petrelfs/mengfanqing/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
processor = CLIPProcessor.from_pretrained("/mnt/petrelfs/mengfanqing/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
 
    return chunks[k]


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


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        

        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        cur_prompt = args.extra_prompt + qs

   

        

        input_ids = preprocess_qwen([{'message': cur_prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
        img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

        image_tensors = []
        for image_file in image_files:
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
        # image_tensors = torch.cat(image_tensors, dim=0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()



def call_llava(image_paths,question):
    
    # Model
    # disable_torch_init()
    # model_path = '/mnt/petrelfs/mengfanqing/.cache/huggingface/hub/models--lmms-lab--llava-next-interleave-qwen-7b-dpo/snapshots/c28ef5b8b62a06292f1c7d94ee009fcf50fdf6a9'
    # model_path = os.path.expanduser(model_path)
    # model_name = get_model_name_from_path(model_path)
    # model_base = None
    # model_name = 'lmms-lab/llava-next-interleave-qwen-7b-dpo'
    # # print(model_name,'model_name')
    # # print(model_path,'model_path')
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    # print(image_processor)
    cur_prompt = question
    conv_mode = "qwen_1_5"
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print('prompt: ',prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    # input_ids = preprocess_qwen([{"from": "human","value": prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
    image_tensors = []
    image_sizes = []
    try:
        for image_path in image_paths:
            image = Image.open(image_path)
            image_sizes.append(image.size)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            # image_tensors.append(image_tensor.half().cuda())
            image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
            image_tensors += image_tensor
    except Exception as e:
        print('image error: ',e)
        return 'image error'

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    try:
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )

        outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        # print(outputs)
        outputs = outputs[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        
        outputs = outputs.strip()
        # print(outputs)
        return outputs
    except:
        print('model error')
        return 'model error'


    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/petrelfs/mengfanqing/.cache/huggingface/hub/models--lmms-lab--llava-next-interleave-qwen-7b-dpo/snapshots/c28ef5b8b62a06292f1c7d94ee009fcf50fdf6a9")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()
    print(args)
    

    modelname = 'Llava-interleave-dpo'
    


    # Model
    disable_torch_init()
    

    model_path = 'modelpath'
    model_path = os.path.expanduser(model_path)
    print('model path: ', model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = None
    model_name = 'llava_qwen'
    
    device_map = "auto"
    # tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=device_map)

    model.eval()

    directory = '/PhyGenBench/PhyVideos/'
   
    modelnames = ['kelingall']
    
    with open('/PhyGenBench/PhyGenBench/explicit_prompts.json','r') as f:
        explicit_prompt = json.load(f)

    for modelname in modelnames:
        num_frames = 32  # 需要采样的帧数
        result = []

        pretrix = "Answer me in Format:{'Choice':'Yes or No','Reason':'the reason'} "
        pretrix2 = "If the Image quality is poor, such as blurry images or noticeable distortion, it should answer No."
        with open(f'/PhyGenBench/PhyGenEval/multi/prompt_replace_augment_multi_question1_{modelname}_res1_imageclip.json','r') as f:
            data = json.load(f)
        video_directory = os.path.join(directory,modelname)
        for i in range(len(data)):
            T2V_prompt = data[i]["caption"]
            Physical_law = data[i]["physical_laws"]
            
            video_path = os.path.join(video_directory,f'output_video_{i+1}.mp4')
            retrieval_question = data[i]['multiimage_question']
            retrieval_prompt = retrieval_question["Retrieval Prompt"]
            question_prompt1 = retrieval_question["Description1"]
            question_prompt2 = retrieval_question["Description2"]
            
            question_prompt3 = explicit_prompt[i]["explicit_caption"]


            output_first_image_path = os.path.join(os.path.join('/PhyGenBench/PhyGenEval/multi/multiimage_clips1',modelname),f'output_video_{i+1}_first.jpg')
            output_middle_image_path = os.path.join(os.path.join('/PhyGenBench/PhyGenEval/multi/multiimage_clips1',modelname),f'output_video_{i+1}_middle.jpg')
            output_last_image_path = os.path.join(os.path.join('/PhyGenBench/PhyGenEval/multi/multiimage_clips1',modelname),f'output_video_{i+1}_last.jpg')

            
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

                question1 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description1}' + f'\n{pretrix}' + f'\n{pretrix2}'

                question2 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description2}' + f'\n{pretrix}'+ f'\n{pretrix2}'

                question3 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description3}' + f'\n{pretrix}'+ f'\n{pretrix2}'




            
            
            
                response1 = call_llava(image_paths1,question1)

                print(response1)

                data[i]["multiimage_question"]["LLava_1"] = response1

                response2 = call_llava(image_paths2,question2)

                data[i]["multiimage_question"]["LLava_2"] = response2


                response3 = call_llava(image_paths3,question3)

                data[i]["multiimage_question"]["LLava_3"] = response3

                result.append(data[i])


            else:
                start_index, max_index, end_index = list(map(int, retrieval_question['CLIP_Index'].split(',')))
                scores_res = []
                cnt1 = 0
                scores1 = []
                frame_middle_output_paths = []
                frame_middle_retrieval_scores = []
                scores3 = []
                for k in range(start_index, max_index+1):
                    
                    # first-retrieval
                    frame_output_path = f"{output_middle_image_path.split('.jpg')[0]}_frame_{k}.jpg"

                    score_re = retrieval_question[f"retrieval_1_{k}"]

                    image_paths1 = [
                        output_first_image_path,frame_output_path
                    ]
                    question1 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description1}' + f'\n{pretrix}'+ f'\n{pretrix2}'
                    response1 = call_llava(image_paths1,question1)
                    data[i]["multiimage_question"][f"LLava_1_{k}"] = response1
                    frame_middle_output_paths.append(frame_output_path)
                    frame_middle_retrieval_scores.append(score_re)


                cnt2 = 0
                scores2 = []
                for m in range(max_index, end_index):
                    
                    frame_output_path = f"{output_middle_image_path.split('.jpg')[0]}_frame_{m}.jpg"

                    score_re2 = retrieval_question[f"retrieval_2_{m}"]


                    image_paths2 = [
                    frame_output_path,output_last_image_path
                    ]

                    
                    question2 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description2}' + f'\n{pretrix}'+ f'\n{pretrix2}'
                    response2 = call_llava(image_paths2,question2)
                    data[i]["multiimage_question"][f"LLava_2_{m}"] = response2

                    if m != max_index:
                        frame_middle_output_paths.append(frame_output_path)
                        frame_middle_retrieval_scores.append(score_re2)

                image_paths3 = [output_first_image_path]
                for tmp in frame_middle_output_paths:
                    image_paths3.append(tmp)
                image_paths3.append(output_last_image_path)
                
                print(len(image_paths3),'len(image_paths3)')
                score_re3 = max(frame_middle_retrieval_scores)
                question3 = 'Look carefully at the picture. Please check if the temporal sequence depicted in these two images matches the following description. First, answer Yes or No, then explain the reason. Think step-by-step.\n' + f'Description: {description3}' + f'\n{pretrix}'+ f'\n{pretrix2}'
                response3 = call_llava(image_paths3,question3)

                data[i]["multiimage_question"]["LLava_3"] = response3
                data[i]["multiimage_question"]["LLava_3_re"] = score_re3

                result.append(data[i])

        print(len(result))
        with open(f'/PhyGenBench/PhyGenEval/multi/prompt_replace_augment_multi_question_{modelname}_res_llava.json','w') as f:
            json.dump(result,f)

    

