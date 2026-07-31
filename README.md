# Task_Handover_HPDS
This repository contains all the scripts of optimized code to the best of my abilities. Tinker around with it and experiment!

## Owner: Aditi Srivastava (AI/ ML Scientist)
Handover To: Pranjal Gupta (Senior AI/ML Scientist)

Scripts in this package:

| Script | Purpose | Dependencies |
| --- | --- | --- |
| finetune_yolo.py | for training on new dataset | Keep dataset and yaml file ready! |
| change_names.py | Daily: Create date-wise datasets of specific train type and split into train, test and val (70:10:20) | Pre-identify the train type using the Alarm json file generateed per train! And run aggregator.py script |
| aggregator.py | For structuring unnamed images in a dataset before. This script is used before changing the names | Same as change_names.py |
| icf_data.yaml | This contains the list of objects for ICF type | NA |
| lhb_data.yaml | This contains the list of objects for LHB type | NA |
| engine_data.yaml | This contains the list of objects for Engine type | NA |
| testing_yolo.py | A rough script for testing a single image - day or night | NA |
| testing_directory.py | Testing the model on the entire test folder | Trained best model |
| pre_processing_techniques.py | This is a rough script used for testing out potential pre-processing strategies  |  |
| missing_image.py | This is a logic to check if there are any missing labels in an image in the dataset | NA |
|  |  |  |

  
## Steps to Run the Code

Step 1: Clone the repository to your system/editor using the command [e.g.: git clone https://github.com/user/repository.git]

Step 2: Create your virtual environment and run [pip install requirements.txt]

Step 3: Map your datasets in the given yaml files as per need. Beware of old file paths in the scripts and change accordingly!

Step 4: run the code as follows: python [name_of_script.py] or python3 [name_of_script.py]

## Existing files included for inference comparison

List of Models:
1. icf_yolov8n.py
2. engine_best.pt
3. lhb_day.pt



