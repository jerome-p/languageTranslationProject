from pathlib import Path

def get_config():
    """
    This function controls the traingin of the model.
    returns a dictinary with all the neccessary parameters like 
    batch_size,num_epochs,learning_rate,seq_length,d_model dimension
    
    Different dataset sources can be mentioned here under 'data source'
    Source and target languages are to be defined in this function
    folder paths for the weights and log files are define under 
    'experiment_name' and 'model_folder' and model's file name defined under 'model_basename'.
    file path to tokenizer is also defined here.
    """
    return {
        "batch_size": 16,
        "num_epochs": 201,
        "lr": 10**-4,
        "seq_len": 150,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "fi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_2"
    }

def get_weights_file_path(config, epoch: str):
    """
    This function returns the model weights file path as a string
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    """
    This function returns the model weights given the epoch number.  
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
