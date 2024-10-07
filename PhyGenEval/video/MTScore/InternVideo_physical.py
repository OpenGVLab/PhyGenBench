import argparse
import numpy as np
import os
import cv2
import json
import torch
from tqdm import tqdm
import time
from configs.config import Config, eval_dict_leaf
from configs.utils import retrieve_text, _frame_from_video, setup_internvideo2

def parse_args():
    parser = argparse.ArgumentParser(description="Video Score Calculation")
    parser.add_argument("--seed", type=int, default=1421538, help="Random seed")
    parser.add_argument('--model_names', nargs='+', default=["test"], help="Name of the models.")
    parser.add_argument("--input_folder", type=str, default="../toy_video", help="Input folder containing videos")
    parser.add_argument("--output_folder", type=str, default="results/all", help="Output folder for saving scores")
    parser.add_argument("--config_path", type=str, default='/PhyGenBench/PhyGenEval/video/MTScore/configs/internvideo2_stage2_config.py', help="Path to config file")
    parser.add_argument("--model_pth", type=str, default='modelpath.pt', help="Path to model checkpoint")
    parser.add_argument('--eval_type', type=int, choices=[150, 1649], default=150)
    return parser.parse_args()

def retry_setup(config, max_attempts=3, delay=2):
    attempts = 0
    while attempts < max_attempts:
        try:
            intern_model, tokenizer = setup_internvideo2(config)
            return intern_model, tokenizer
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            if attempts == max_attempts:
                raise Exception("All attempts failed.")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

def calculate_video_score(video_path, text_to_index, text_candidates, intern_model, config):
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]

    texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=4, config=config)


    return texts, probs

def calculate_average_score(scores):
    total_videos = len(scores)
    total_general_score = sum(score[0] for score in scores)
    total_metamorphic_score = sum(score[1] for score in scores)
    average_general_score = total_general_score / total_videos
    average_metamorphic_score = total_metamorphic_score / total_videos
    return average_general_score, average_metamorphic_score

def load_existing_scores(filepath):
    with open(filepath, 'r') as file:
        scores = json.load(file)
    return scores

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        config = Config.from_file(args.config_path)
    except:
        config = Config.from_file(args.config_path.replace("configs/", "MTScore/configs/"))

    config = eval_dict_leaf(config)

    config['model']['vision_encoder']['pretrained'] = args.model_pth

    intern_model, tokenizer = retry_setup(config, 10, 2)


    modelnames = ['kelingall']
    
    for modelname in modelnames:
        with open('/PhyGenBench/PhyGenBench/video_question.json','r') as f:
            data = json.load(f)


        result = []
        for i in range(len(data)):
            data_tmp = data[i]
            text_candidates = [
                f"Completely Fantastical:{data_tmp['video_question']['Description1']}",
                f"Highly Unrealistic:{data_tmp['video_question']['Description2']}",
                f"Slightly Unrealistic:{data_tmp['video_question']['Description3']}",
                f"Almost Realistic:{data_tmp['video_question']['Description4']}"
                
            ]

            text_to_index = {text: index for index, text in enumerate(text_candidates)}

            video_path = f'/PhyGenBench/PhyVideos/{modelname}/output_video_{i+1}.mp4'
            texts, probs = calculate_video_score(video_path, text_to_index, text_candidates, intern_model, config)

            print(texts,probs)
            # break
            choice = texts[0].split(':')[0]

            if 'Completely Fantastical' in choice:
                score = 0
            elif 'Highly Unrealistic' in choice:
                score = 1
            elif 'Slightly Unrealistic' in choice:
                score = 2
            elif 'Almost Realistic' in choice:
                score = 3
            
            data_tmp['InternVideo2_score'] = score

            result.append(data_tmp)
            print(i)


        print(len(result))
        with open(f'/PhyGenBench/PhyGenEval/video/prompt_replace_augment_video_question_{modelname}_res_intern.json','w') as f:
            json.dump(result,f)

    

if __name__ == "__main__":
    main()