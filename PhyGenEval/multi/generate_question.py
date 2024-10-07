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
    ### Task Instructions for a Physical Laws Judge
    You are a seasoned physical laws judge. You will receive a T2V Prompt along with the primary physical law it represents. Your task is to:

    1. **Understand the physical process** described in the prompt.
    2. **Determine whether the process is monotonic** (i.e., a process that changes in a single direction without reversal).
    - If monotonic: Use the default retrieval prompt: "Middle Frame."
    - If non-monotonic (i.e., processes with reversals or dynamic changes): Provide a custom retrieval prompt based on a significant event or change within the process.
    3. For **both types of processes**, provide two ideal descriptions based on the chosen frame pairs (first-retrieval, retrieval-last, and first-last). The descriptions should focus on the physical changes that should occur between the frames and It must be a simple dynamic process. Focus on the objects that have changed/remained unchanged in the second image relative to the first image.
    4. **Ensure the descriptions are clear, simple, and accurate**, emphasizing the transitions between frames.
    5. **Ensure simplicity and clarity in the descriptions**, with focus on the physical phenomena caused by the behavior in the prompt, rather than the behavior itself. For example, when a piece of chalk slides on a blackboard, don't worry about the chalk, but whether the trace drawn by the chalk becomes longer.


    ### Examples

    #### Example 1

    **Input:**

    - **Prompt:** A hand slowly presses a needle into the surface of a fully inflated balloon, with the tip gradually penetrating the outer layer.
    - **Physical Law:** Air pressure

    **Output:**

    {
    "Retrieval Prompt": "Needle inserted into the balloon",
    "Description1": "the balloon does not explode.",
    "Description2": "the balloon explodes."
    }

    #### Example 2

    **Input:**

    - **Prompt:** A delicate, fragile egg is hurled with significant force towards a rugged, solid rock surface, where it collides upon impact.
    - **Physical Law:** The stone is harder than the egg

    **Output:**

    {
    "Retrieval Prompt": "The egg contacts the rock surface",
    "Description1": "the egg does not break, and the rock remains unchanged.",
    "Description2": "the egg breaks, but the rock remains unchanged."
    }


    #### Example 3

    **Input:**

    - **Prompt:** A timelapse of air being gradually and forcefully extracted from the mouth of an empty, thin, sealed plastic bottle.
    - **Physical Law:** Atmospheric pressure

    **Output:**

    {
    "Retrieval Prompt": "Middle Frame",
    "Description1": "The bottle is shrinking.",
    "Description2": "The bottle is shrinking."
    }

    ### Note
    Ensure that the retrieval prompts, frame descriptions, and adherence to physical laws are formulated clearly and correctly. Monotonic processes should use the default "Middle Frame" retrieval prompt. The description does not need to explain the reasons, only focus on the phenomena. The description should emphasize the changes in the latter image compared to the previous one and it must be a simple dynamic process, with focus on the physical phenomena caused by the behavior in the prompt, rather than the behavior itself.

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

    

    # parsed_data = ast.literal_eval(input_string.strip())
    input_string = input_string.strip().lstrip("```json").rstrip("```").strip()
    # parsed_data = json.loads(input_string.strip())

    try:
        parsed_data = json.loads(input_string)
    except:
        print('except')
        parsed_data = input_string
    
    


    # Output the parsed data
    print(parsed_data)

    data_tmp['multiimage_question'] = parsed_data
    result.append(data_tmp)


    # break
print(len(result))

with open('/PhyGenBench/PhyGenBench/multi_question.json','w') as f:
    json.dump(result,f)
