import os
from pathlib import Path

import torch
from aion.model import AION
from aion.codecs import CodecManager
from aion.modalities import Z
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq

from data.load_data import LegacyMultimodalDataModule, format_data_modalities

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
DATA_FOLDER = os.getenv("DATA_FOLDER")
base = Path(DATA_ROOT) / DATA_FOLDER

combinations = {
    "true_z": ["Z_HP"], # perhaps this should not go through the model. Let's just discretize ourself.
    "photometry": ["g", "r", "i", "z"],
    "imaging": ["image"],
    "spectroscopy": ["spectrum"],
    "image_photometry": ["image", "g", "r", "i", "z"],
    "image_spectrum": ["image", "spectrum"],
    "image_spectrum_photometry": ["image", "spectrum", "g", "r", "i", "z"],
    "image_spectrum_photometry_true_z": ["image", "spectrum", "g", "r", "i", "z", "Z_HP"],
}

# Could also do combinations of photometry. A bit verbose.

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    with torch.no_grad():
        codec_manager = CodecManager(device="cuda")

        model = AION.from_pretrained(
            "polymathic-ai/aion-base"
        ).to("cuda").eval()
        data = LegacyMultimodalDataModule(
            batch_size=128,
            num_workers=1
        )
        data.setup()
        dataloader = data.dataloader()

        for idx, batch in enumerate(dataloader):
            batch = format_data_modalities(batch)
            # Pop keys and ** in the model for each iteration of comparisons.
            for name, modalities in combinations.items():
                tok_z = model(
                    codec_manager.encode(
                        *[batch[modality] for modality in modalities]
                    ),
                    target_modality=Z
                )["tok_z"].detach().cpu().squeeze(1).numpy()

                table = pa.Table.from_pydict(
                    {
                        "TARGETID": batch["TARGETID"].numpy(),
                        "tok_z": pa.array(tok_z.tolist(), type=pa.list_(pa.float32())),
                        "Z_HP": batch["Z_HP"].value.detach().cpu().numpy()
                    }
                )

                fp = base / f"{name}_1.0"
                if not fp.exists():
                    fp.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, fp / f"{idx}.parquet")
