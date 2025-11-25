import os
from pathlib import Path

from data.load_tokens import DATA_FILES, combinations
from datasets import load_dataset
import torch 
import torch.nn.functional as F
from dotenv import load_dotenv

data = {}
divs = {}

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
DATA_FOLDER = os.getenv("DATA_FOLDER")
base = Path(DATA_ROOT) / DATA_FOLDER

for name in combinations.keys():
    data[name] = load_dataset("parquet", data_files=DATA_FILES, split=name)

combinations.pop("true_z")


true_z = torch.tensor(data["true_z"]["tok_z"])
true_z_p = F.softmax(true_z, dim=-1)

for name in combinations.keys():
    tok_z = torch.tensor(data[name]["tok_z"])
    tok_z_p = F.softmax(tok_z, dim=-1)

    ce = torch.sum(
        -true_z_p * torch.log(tok_z_p),
        dim=-1
    )

    divs[name] = ce

torch.save(divs, base / "ce.pt")