import pandas as pd 
import yaml
from pathlib import Path 

def load_config(config_path : str) -> dict : 
    with open(config_path, "r", encoding="utf-8") as f : 
        config = yaml.safe_load(f)
    return config

def clean_data (df : pd.DataFrame) : 
    df = df.drop_duplicates()
    return df 



