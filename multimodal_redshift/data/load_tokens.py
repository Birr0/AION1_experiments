import os

from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
DATA_FOLDER = os.getenv("DATA_FOLDER")
base = Path(DATA_ROOT) / DATA_FOLDER


combinations = {
    "true_z": ["Z_HP"],
    "photometry": ["g", "r", "i", "z"],
    "imaging": ["image"],
    "spectroscopy": ["spectrum"],
    "image_photometry": ["image", "g", "r", "i", "z"],
    "image_spectrum": ["image", "spectrum"],
    "image_spectrum_photometry": ["image", "spectrum", "g", "r", "i", "z"],
    "image_spectrum_photometry_true_z": ["image", "spectrum", "g", "r", "i", "z", "Z_HP"],
}

DATA_FILES = {
    name: str(base / name / "*.parquet")
    for name in combinations.keys()
}


datasets_dict = load_dataset("parquet", data_files=DATA_FILES, split="true_z")
print(len(datasets_dict["tok_z"][0]))
print(len(datasets_dict["TARGETID"]))

'''print(datasets_dict[0]["true_z"])
print(datasets_dict[0]["TARGETID"])'''