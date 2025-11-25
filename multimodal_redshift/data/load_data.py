import os 

from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from astropy.table import Table
from dotenv import load_dotenv
import numpy as np
import lightning as L

from aion.modalities import (
    LegacySurveyImage,
    DESISpectrum,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
    Z,
)

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
DATA_FOLDER = os.getenv("DATA_FOLDER")
FILE_NAME = "provabgs_legacysurvey_eval_v2.fits"
FILE_PATH = os.path.join(DATA_ROOT, DATA_FOLDER, FILE_NAME)

def load_data():
    return Table.read(FILE_PATH)

def format_data_modalities(batch, device="cuda"):
    """Formats the input data into modality objects."""

    # Helper function
    def to_tensor(data_array, dtype="float32"):
        return torch.tensor(np.array(data_array).astype(dtype), device=device)

    # Create image modality
    image = LegacySurveyImage(
        flux=to_tensor(batch["legacysurvey_image_flux"]),
        bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
    )

    # Create spectrum modality
    spectrum = DESISpectrum(
        flux=to_tensor(batch["desi_spectrum_flux"]),
        ivar=to_tensor(batch["desi_spectrum_ivar"]),
        mask=to_tensor(batch["desi_spectrum_mask"], dtype="bool"),
        wavelength=to_tensor(batch["desi_spectrum_lambda"]),
    )

    # Create flux modalities
    g = LegacySurveyFluxG(value=to_tensor(batch["legacysurvey_FLUX_G"]))
    r = LegacySurveyFluxR(value=to_tensor(batch["legacysurvey_FLUX_R"]))
    i = LegacySurveyFluxI(value=to_tensor(batch["legacysurvey_FLUX_I"]))
    z = LegacySurveyFluxZ(value=to_tensor(batch["legacysurvey_FLUX_Z"]))

    Z_HP = Z(to_tensor(batch["Z_HP"]))

    return {
        "image": image, 
        "spectrum": spectrum,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "Z_HP": Z_HP,
        "TARGETID": batch["TARGETID"],
    }


class LegacyMultimodalDataset(Dataset):
    def __init__(self):
        self.data = load_data()
        self.cols = [
            "legacysurvey_image_flux",
            "desi_spectrum_flux",
            "desi_spectrum_ivar",
            "desi_spectrum_mask",
            "desi_spectrum_lambda",
            "legacysurvey_FLUX_G",
            "legacysurvey_FLUX_R",
            "legacysurvey_FLUX_I",
            "legacysurvey_FLUX_Z",
            "Z_HP"
        ]

    @staticmethod
    def to_tensor(data_array, dtype="float32"):
        return torch.tensor(np.array(data_array).astype(dtype))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            col: self.to_tensor(self.data[idx][col]) for col in self.cols
        }
        item["TARGETID"] = self.data[idx]["TARGETID"]
        return item

class LegacyMultimodalDataModule(L.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.dataset = LegacyMultimodalDataset()
    
    def dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

if __name__ == "__main__":
    #data = load_data()
    #image, spectrum, g, r, i, z = format_data_modalities(data)

    '''data = LegacyMultimodalDataModule(
        batch_size=128,
        num_workers=1,
    )
    data.setup()
    '''
    data = load_data()
    print(data)
    print(data.colnames)