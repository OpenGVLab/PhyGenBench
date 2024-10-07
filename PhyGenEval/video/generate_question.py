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
import re

client = OpenAI(
    base_url='',
    api_key='',
)


with open('/PhyGenBench/PhyGenBench/prompts.json','r') as f:
    data = json.load(f)

result = []

for i in range(len(data)):
    data_tmp = data[i]
    prompt = data_tmp['caption']
    physical_law = data_tmp['physical_laws']

    GPT_generate_question_prompt = """
    **Task Instruction for a Seasoned Physical Laws Judge**

    **Task Overview:**
    You will evaluate the realism of a T2V (Text-to-Video) prompt based on the given physical law it represents. Your primary focus is on how well the video's temporal sequence matches the ideal physical process described by the law. You will assess the video across five levels of realism: "Completely Fantastical," "Clearly Unrealistic," "Moderately Unrealistic," "Slightly Unrealistic," and "Almost Realistic."

    **Evaluation Standards:**
    1. **Completely Fantastical:** Displays complete detachment from reality throughout, with elements of fantasy or surrealism.
    2. **Clearly Unrealistic:** Contains significant distortions over extended periods or on a large scale, making the overall scene unrealistic or contrary to physical laws, such as unrealistic large objects or scenes.
    3. **Slightly Unrealistic:** Distortions are brief or minute, hard to notice, such as unnatural facial expressions or unnatural scene textures.
    4. **Almost Realistic:** No noticeable distortions; aligns completely with reality.


    **Expected Output Format:**
    You will only provide descriptions for each level of realism using the following format and do not include other texts:
    {
    'Description1':'The content of Completely Fantastical description',
    'Description2':'The content of Clearly Unrealistic description',
    'Description3':'The content of Slightly Unrealistic description',
    'Description4':'The content of Almost Realistic description',
    }

    ---

    Please generate the corresponding descriptions for the given T2V prompt and physical law.

    Here is the input:
    Prompt:An elastic rubber ball is forcefully thrown onto the floor, impacting the surface with considerable speed.
    Physical Law:Elasticity

    ### Note
    You need to pay attention to the clear distinction between descriptions of different levels, so that you can easily determine which level different videos belong to and thus determine their correctness in the physical process. In addition, the generated descriptions of different levels need to have strong timing characteristics and pay attention to the characteristics of the video. Ensure the response is valid JSON.

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
                "content": json.dumps({"type": "text", "text": GPT_generate_question_prompt}),
            }
        ],
        max_tokens=512,
    )
    print(response.choices[0].message.content)

    input_string = response.choices[0].message.content

    # input_string = re.sub(r"(?<!\\)'", '"', input_string)

    input_string = input_string.strip().lstrip("```json").rstrip("```").strip()
    # parsed_data = json.loads(input_string.strip())

    parsed_data = json.loads(input_string)
    # parsed_data = ast.literal_eval(input_string.strip())

    # Output the parsed data
    print(parsed_data)

    data_tmp['video_question'] = parsed_data
    result.append(data_tmp)


    # break
print(len(result))

with open('/PhyGenBench/PhyGenBench/video_question.json','w') as f:
    json.dump(result,f)




