import time
import json
import glob
import torch  
import contextlib
from pathlib import Path
from openai import OpenAI
import utils  
import configparser
import logger
import logging
import io, os, sys


"""
The code is partly based on the code by Maris Buttmann from the paper:
Appeal, Align, Divide? Stance Detection for Group-Directed Messages in German Parliamentary Debates. LREC 2026.
"""

def get_config() -> str:
    """Gets the user's Huggingface API key from a config file."""
    path = Path('.') / "secrets.json"
    try:
        with open(path, "r") as config_file:
            config = json.load(config_file)  
        return config 
    except FileNotFoundError:
        logging.info("Error: secrets file not found at {path}")
        return None

"""
Takes a list of helper_dicts, an openAI client
[{
    "token_id": idx,
    "prompt_type":prompt_type,
    "message": message
    }, 
    ...]
"""
def batch_inference(
    prompt_batch: list[dict],
    client: OpenAI, 
    model_name: str,
    technique: str
) -> list[dict]:

    # Prepare all formatted prompt strings in a list   
    logging.info("Generating responses for {} prompts in batch...".format(len(prompt_batch))) 
    for i in range(len(prompt_batch)):
        prompt_dict = prompt_batch[i]           
        logging.info("Processing correction no. {}".format(i))
        answer = client.chat.completions.create(
            temperature = 0,
            messages=prompt_batch[i]["message"],  
            model=model_name,
        )
        result = {
            "content": answer.choices[0].message.content,
            "model": answer.model,
            "usage": answer.usage.model_dump()
        } 
        prompt_batch[i]["prediction"] = result["content"]
        
    logging.info("Return records with length {}".format(len(prompt_batch)))
    return prompt_batch


"""
Filter data for different error types
"""
def filter_jsonl_data(jsonl_data, error_type):
    if error_type == "NN_NE":
        err_tags = ['NN', 'NE']
        for item in jsonl_data:
            item['error_ids'] = [i for i in range(len(item['token'])) if item['error'][i] in err_tags and item['pos'][i] in err_tags]
            item['predictions'] = []

    elif  error_type == "FM_OTHER": 
        err_tags = ['FM']
        for item in jsonl_data:
            item['error_ids'] = [i for i in range(len(item['token'])) if item['error'][i] != 'OK' and item['pos'][i] in err_tags]
            item['predictions'] = []

    elif error_type == "VVFIN_OTHER": 
        err_tags = ['VVFIN']
        for item in jsonl_data:
            item['error_ids'] = [i for i in range(len(item['token'])) if item['error'][i] != 'OK' and item['pos'][i] in err_tags]
            item['predictions'] = [] 

    elif error_type == "PTKZU_OTHER": 
        err_tags = ['PTKZU']
        for item in jsonl_data:
            item['error_ids'] = [i for i in range(len(item['token'])) if item['error'][i] != 'OK' and item['pos'][i] in err_tags]
            item['predictions'] = [] 

    elif error_type == "ADV_OTHER": 
        err_tags = ['ADV']
        for item in jsonl_data:
            item['error_ids'] = [i for i in range(len(item['token'])) if item['error'][i] != 'OK' and item['pos'][i] in err_tags]
            item['predictions'] = [] 

    elif error_type == "GENERIC":  
        for item in jsonl_data:
            item['error_ids'] = [i for i in range(len(item['token'])) if item['error'][i] != 'OK' and item['pos'][i] != item['error'][i]]
            item['predictions'] = [] 



def process_test_set(
    client: OpenAI, 
    model_name: str, 
    dataset_dir: str, 
    result_dir: str, 
    prompt_type:str):
    
    # # Get the pos tagged data that needs to be corrected
    inpath  = dataset_dir + '*.tsv' 

    logging.info("Reading input from {}".format(inpath))
    infiles = glob.glob(inpath)
    logging.info(infiles)

    for f in infiles:
        logging.info("Reading file {}".format(f))
        # read input data
        jsonl_data = utils.get_test_data(f)  
        prompt_list = [prompt_type]

        # 1. Process prompts
        for prompt_type in prompt_list:
            outpath = result_dir + prompt_type + '_' + f.split('/')[-1]
            logging.info("Processing {}".format(prompt_type))

            # Filter the data according to our error type
            filter_jsonl_data(jsonl_data, prompt_type) 

            for i in range(len(jsonl_data)): 
                # Get batch of to-be-processed prompts as a list of helper dicts
                # [{
                #   "token_id": idx,
                #   "prompt_type":prompt_type,
                #   "message": message
                # }, ...]
                # Skip if sentence does not contain an error of error_type
                if len(jsonl_data[i]['error_ids']) == 0:
                    continue

                prompt_batch = utils.get_test_batch(jsonl_data[i], prompt_type)

                logging.info("Batch len: {}".format(len(prompt_batch)))
                prompt_batch = batch_inference(prompt_batch, client, model_name, technique=prompt_type)

                jsonl_data[i]["predictions"] = [item["prediction"] for item in prompt_batch]

            logging.info("Writing records to: {}".format(outpath))
            utils.write_to_file(jsonl_data, outpath)


def main():
    """
    Main function to load the LLM and start the predictions.
    """    
    # Read config file
    prompt_type = config_parser['BASE']['prompt_type'] 
    model_name  = config_parser['MODEL']['model_name']
    dataset_dir = config_parser['DATA']['dataset_dir']
    result_dir = config_parser['DATA']['result_dir'] 

    # Get api key and url for LLM access point
    config = get_config()
    api_key = config["api_key"]
    base_url = config["base_url"]
    hf_home  = config["hf_home"]

    if hf_home:
        os.environ["HF_HOME"] = hf_home
    else:
        logging.info("Hugging Face Home not found.")
    
    # Get the number of available GPUs    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logging.info("Device is {}. Found {} GPUs.".format(device, num_gpus))

    start_time = time.time() 
    
    logging.info("Initialising model: {}".format(model_name))
    logging.info("This may take several minutes...")

    # Initialise the OpenAI client
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(e)
        logging.info("Error during model initialisation:")  
        return

    elapsed_time = (time.time() - start_time) / 60
    logging.info("--- LLM setup complete in {elapsed_time:.2f} min. ---")

    # Start processing the test data
    process_test_set(client, model_name, dataset_dir, result_dir, prompt_type)
    


if __name__ == "__main__":

    config_parser = configparser.ConfigParser()
    config_parser.read(sys.argv[1])

    logging.basicConfig(level=logging.INFO)

    logging.info("Starting main...")
    main()


"""
cd /ceph/inrehbei/proj/huggingface_hub/hub

# update python to 3.13

pip install gpt-oss

python -m gpt_oss.chat model/
"""
