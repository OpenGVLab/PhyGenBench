# Video Evaluation Tool

This is a tool for evaluating the quality of generated videos, primarily based on GPT-4 and computer vision technologies.

## Requirements

- Python 3.9
- Other dependencies (see below)

## Installation

1. Ensure Python 3.9 is installed on your system. You can download and install it from the [Python official website](https://www.python.org/downloads/).

2. Clone this repository to your local machine:

   ```
   git clone [repository URL]
   cd [repository directory]
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```
   conda create -n PhyGenbench python=3.9
   ```

4. Install the required dependencies:

   ```
   pip install openai argparse pillow opencv-python numpy tqdm
   ```

## Usage

1. Prepare your OpenAI API key.

2. Run the main script, for example:

   ```
   python gpt4o_sementic.py --openai_api_key YOUR_API_KEY --directory /path/to/videos --modelnames model1 model2 --output_dir /path/to/output
   ```

   Replace `YOUR_API_KEY` with your actual OpenAI API key, and adjust other parameters as needed.

3. Key parameters:
   - `--openai_api_key`: Your OpenAI API key (required)
   - `--directory`: Directory containing video files
   - `--modelnames`: Names of models to evaluate (can specify multiple, ['kling', 'cogvideo5b'])
   - `--output_dir`: Directory for output results
   - `--num_frames`: Number of frames to extract from each video (default 16)
   - `--gpt_augment_eval`: Path to JSON file containing evaluation data

4. The script will process videos in the specified directory and generate CSV files with evaluation results in the output directory.

python gpt4o_sementic.py --openai_api_key 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 --directory /Users/ljq/Downloads/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/PhyBench-Videos --modelnames 'keling' --output_dir /Users/ljq/Downloads/