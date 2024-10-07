import openai
import argparse
import os
import json
import base64
import csv
import time
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from openai import OpenAI

# Set up OpenAI API client
def setup_openai_client(api_key, base_url='https://api.openai.com/v1'):
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

def call_gpt4(prompt, frames=None):
    # 如果 frames 存在，将它们添加到消息中
    messages = [{"role": "system", "content": prompt},
                {"role":"user", "content":[
          "These are the frames from the video.",
          *map(lambda x:{"type":"image_url",
                         "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":"high"}}, frames)
        ],
      }]
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def extract_json(string):
    start = string.find('{')
    end = string.rfind('}') + 1
    json_part = string[start:end]
    
    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        print(string)
        print("Invalid JSON part")
        return None

def process_video(video_path, num_frames=16):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算需要跳过的帧数，确保均匀分布
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if not success:
            break
        frame = cv2.resize(frame, (720, 480))
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

def call_llava(sampled_frames, sementic):
    # 生成图像描述的 prompt
    action = sementic["action"]
    object1 = sementic["object1"]
    object2 = sementic.get("object2","")

    if object2 != "":
        object_string = f"{object1} and {object2}"
    else:
        object_string = f"{object1}"



    # 生成第二个 prompt，询问是否图像准确展示了给定的caption
    Q1 = f"According to the key images from the video, evaluate if {object_string} is presented in the video. Assign a score from 1 to 3 according to the criteria:\n \
3: All the objects are present.\n \
2: Some of the objects involved in the interaction are missing.\n \
1: None of the objects involved in the interaction are present.\n \
Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2) and then explain it, step by step.\n \
Note: You do not need to consider whether the physical phenomena described are scientifically accurate. Only focus on whether the object is presented in the images."
    print(len(sampled_frames))

    outputs_1 = call_gpt4(Q1, sampled_frames)

    outputs_2 = "none"

    # 提取 JSON 数据
    json_obj = extract_json(outputs_1)

    json_obj_2 = {
        "score": 0,
        "explanation": "no action"
    }

    time.sleep(3)
    try:
        score_tmp = json_obj["score"]
    except:
        print(json_obj)
        score_tmp = "bad reply"

    if score_tmp <3:
        total_score = score_tmp
    else:
        total_score = 3
        if action == "no action":
            total_score = 4
            json_obj_2 = {
            "score": 1,
            "explanation": "no action"
            }
        else:
            Q2 = f"According to the key images from the video, evaluate if {action} is presented in the video. Assign a score from 0 to 1 according to the criteria:\n \
            1: the action is presented in this video.\n \
            0: the action is not presented in this video.\n \
            Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2) and then explain it, step by step."

            outputs_2 = call_gpt4(Q2, sampled_frames)

            json_obj_2 = extract_json(outputs_2)

            total_score += json_obj_2["score"]

    return "test", json_obj, json_obj_2, total_score

def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames
    
def model_score(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1])-1)/3  # normalize
                score += score_tmp
                cnt += 1
            except:
                continue
        
        score = score / cnt
        print("Number of images evaluated:", cnt, "Object interactions model score:", score)
        
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["score:", score]) 
        
def get_csv_length(csv_path):
    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            print(len(lines))
            return len(lines)  # Return the number of rows in the CSV file
    return 0  # If the file doesn't exist, return 0

# Main function entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample from each video")
    parser.add_argument("--directory", type=str, default='/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/PhyBench-Videos/', help="Directory containing video folders")
    parser.add_argument("--modelnames", nargs='+', default=['cogvideo5ball_fix'], help="List of model names to process")
    parser.add_argument("--gpt_augment_eval", type=str, default='GPT_agument_eval.json', help="Path to GPT augment evaluation JSON file")
    parser.add_argument("--output_dir", type=str, default="/mnt/petrelfs/mengfanqing/codebase_ljq/Video_eval/Results/", help="Output directory for results")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    args = parser.parse_args()

    print(args.modelnames)

    # Set up OpenAI client
    client = setup_openai_client(args.openai_api_key)
    result = []

    with open(args.gpt_augment_eval, 'r') as f:
        data = json.load(f)

    for modelname in args.modelnames:
        csv_path = os.path.join(args.output_dir, f'{modelname}/object_interactions_score_gpt4o.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Get the length of existing CSV file (number of rows)
        existing_length = get_csv_length(csv_path)
        print(f"Existing length of CSV: {existing_length} rows")

        # Open file in append mode
        with open(csv_path, 'a', newline='') as csvfile: 
            csv_writer = csv.writer(csvfile)

            # Write header if file doesn't exist or has no data
            if existing_length == 0:
                csv_writer.writerow(["prompt", "object_score", "object_reason", "action_score", "action_reason", "total_Score"])

            video_directory = os.path.join(args.directory, modelname)
            score_total = 0

            # Process data starting from existing length to avoid duplication
            for i in range(existing_length-1, len(data)):
                T2V_prompt = data[i]["caption"]
                Physical_law = data[i]["physical_laws"]
                sementic_object = data[i]["sementic"]

                video_path = os.path.join(video_directory, f'output_video_{i+1}.mp4')

                total_frames = get_video_frame_count(video_path)
                print(f"Total frames in video: {total_frames}")

                sampled_frames = process_video(video_path, args.num_frames)

                test, outputs_1, outputs_2, score_tmp = call_llava(sampled_frames, sementic_object)

                object_score = outputs_1["score"]
                object_reason = outputs_1["explanation"]

                action_score = outputs_2["score"]
                action_reason = outputs_2["explanation"]

                score_total += score_tmp
                csv_writer.writerow([T2V_prompt, object_score, object_reason, action_score, action_reason, score_tmp])
                csvfile.flush()

        model_score(csv_path)