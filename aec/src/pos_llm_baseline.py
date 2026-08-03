import time
import json
import glob
import torch  
import contextlib
from pathlib import Path
from openai import OpenAI
from ast import literal_eval
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
Takes a list of helper_dicts and an openAI client
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
        
        logging.info("Processing instance no. {}".format(i))
        answer = client.chat.completions.create(
            temperature = 0,
            messages=prompt_batch[i]["message"],  
            model=model_name,
        )
        result = {
            "content": literal_eval(answer.choices[0].message.content),
            "model": answer.model,
            "usage": answer.usage.model_dump()
        }        
        prompt_batch[i]["prediction"] = result["content"]["predictions"]
    
    logging.info("Return records with length {}".format(len(prompt_batch)))
    return prompt_batch



def process_test_set(
    client: OpenAI, 
    model_name: str, 
    dataset_dir: str, 
    result_dir: str, 
    prompt_type:str):
    # Get the pos tagged data that needs to be corrected
    inpath  = dataset_dir + '*.tsv' 

    logging.info("Reading input from {}".format(inpath))
    infiles = glob.glob(inpath)
    logging.info(infiles)

    for f in infiles:

        logging.info("Reading file {}".format(f))
        # read input data
        jsonl_data = utils.get_test_data(f)  
        # add prompt type to list
        prompt_list = [prompt_type] 
        
        for prompt_type in prompt_list:
            outpath = result_dir + prompt_type + '_' + f.split('/')[-1]
            logging.info("Processing {}".format(prompt_type))

            # we want to predict pos labels for each token
            for i, item in enumerate(jsonl_data): 
                prompt_batch = utils.get_baseline_batch(item, prompt_type)
                prompt_batch = batch_inference(prompt_batch, client, model_name, technique=prompt_type)
                jsonl_data[i]["predictions"] = prompt_batch[0]["prediction"]

                # If you want to save predictions to file after every 25th call
                # to prevent data loss (in case of an unstable connection),
                # uncomment the next two lines. 
                #if i%25 == 0:
                #    utils.write_to_file(jsonl_data, outpath+str(i))

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
    # Initialize the OpenAI client
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

    logging.info("Starting baseline predictions...")
    main()


