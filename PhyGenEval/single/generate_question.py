import cv2
from moviepy.editor import VideoFileClip
import time
import base64
import os
import json
import base64
import requests
import jsonlines
from multiprocessing import Pool
from openai import OpenAI
from functools import partial
import time
from IPython.display import Image, display, Audio, Markdown
import ast


client = OpenAI(
    base_url='',
    api_key='',
)



with open('/PhyGenBench/PhyGenBench/prompts.json','r') as f:
    data = json.load(f)

result = []
for i in range(len(data)):
    data_tmp = data[i]
    prompt = data_tmp["caption"]
    physical_law = data_tmp["physical_laws"]

    GPT_generate_question_prompt = """
    Now You are a "Physical Laws Judger", Read this instruction below and generate the output for the input: Task Overview: You will receive a T2V prompt along with a specific physical law to be judged. Based on these, propose 1-2 questions for the T2V prompt, with Yes/No answers and the statement using question and answer. If it can produce different problems, then use 2, otherwise use 1. Question Requirements: Image-Based: Questions must be based on a single image. Retrieval Prompt: If this single image appears in the middle of the video, It should include a retrieval prompt for locating the image, else, it can use the last frame (see next note). Special Note on Last Frame: If the question can be asked using the last frame of the video, simply state "Last Frame" instead of providing a retrieval prompt. Diversity of Questions: Note: The two questions must be different from each other. Note: Ensure the questions are easy to judge based on real-world physical conditions. Input Format: Prompt: [Description of the scene or event] Physical Law: [Statement of the physical law that should be judged] Expected Output: Retrieval Prompt: prompt / Last Frame Question: Simple, based on a single image, and different for each question Answer: Yes/No, and Statement

    Here is some examples:
    Input:
    Prompt:A hand slowly presses a needle into the surface of a fully inflated balloon, with the tip gradually penetrating the outer layer.
    Physical Law:Air pressure

    Output:
    [
        {
            'Retrieval Prompt':'Needle inserted into the balloon',
            'Question':'Does the balloon explode?',
            'Answer':'Yes',
            'Statement':'The balloon explodes',
            'Antonym':'The ballon remains intact
        }
    ]

    Input:
    Prompt:A delicate, fragile egg is hurled with significant force towards a rugged, solid rock surface, where it collides upon impact 
    Physical Law:The stone is harder than the egg

    Output:

    [

    {
    'Retrieval Prompt':'The egg contacts the rock surface',
    'Question':'Does the egg shows visible signs of cracking or deformation?',
    'Answer':'Yes',
    'Statement':'The egg shows visible signs of cracking or deformation',
    'Antonym':'The egg remains intact

    },
    {
    'Retrieval Prompt':'Last Frame',
    'Question':'Does the rock remain intact?',
    'Answer':'Yes',
    'Statement':'The rock remain intact',
    'Antonym':'The glass bottle remains intact
    }
    ]
    """

    input_prompt = f"""
    Here is the input:
    Prompt: {prompt}
    Physical Law: {physical_law}
    """

    GPT_generate_question_prompt = GPT_generate_question_prompt + input_prompt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GPT_generate_question_prompt},
                ],
            }
        ],
        max_tokens=512,
    )
    print(response.choices[0].message.content)

    input_string = response.choices[0].message.content

    input_string = input_string.strip().lstrip("```json").rstrip("```").strip()
    # parsed_data = json.loads(input_string.strip())

    try:
        parsed_data = json.loads(input_string)
    except:
        print('except')
        parsed_data = input_string
    
    


    # Output the parsed data
    print(parsed_data)

    data_tmp['singleimage_question'] = parsed_data
    result.append(data_tmp)

print(len(result))

with open('/PhyGenBench/PhyGenBench/single_question.json','w') as f:
    json.dump(result,f)
