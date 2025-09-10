from representation_learning.models.perch import PerchModel
import torch
import numpy as np
import pandas as pd
from zeroshot_detection_eval.data.dataloader import DataloaderBuilder
from esp_data.io import filesystem
import os 
from tqdm import tqdm

DATASET = "powdermill"
BATCH_SIZE = 1
WINDOW_SIZE = 5
N_WORKERS = 8
ANNOTATION_COLUMN = "Species"

out_fp = f"perch_{DATASET}_{WINDOW_SIZE}_{ANNOTATION_COLUMN}.yaml"

model = PerchModel(0)
dl = DataloaderBuilder("powdermill", WINDOW_SIZE, WINDOW_SIZE, BATCH_SIZE, N_WORKERS)

# convert output labels ebird -> scientific name
output_labels = pd.read_csv('perch_labels.csv')['ebird2021_code'].tolist()

info_fp = "Clements-v2024-October-2024-rev.csv"
if not os.path.exists(info_fp):
    fs = filesystem("gcs")
    gcp_path = "gs://esp-ml-datasets/wabad/v0.1.0/Clements-v2024-October-2024-rev.csv"
    fs.get(gcp_path, info_fp)

info_df = pd.read_csv(info_fp)
info_df = info_df[~pd.isna(info_df["scientific name"])]

output_species_labels = []
for ebird_label in tqdm(output_labels):
    df_sub = info_df[info_df['species_code'] == ebird_label]
    if DATASET == "powdermill":
        if ebird_label == "haiwoo":
            scientific_name = 'Leuconotopicus villosus'
            output_species_labels.append(scientific_name)
            continue
            
        if ebird_label == "dowwoo":
            scientific_name = 'Picoides pubescens'
            output_species_labels.append(scientific_name)
            continue

        if ebird_label == "ruckin":
            scientific_name = 'Regulus calendula'
            output_species_labels.append(scientific_name)
            continue

        if ebird_label == "reevir":
            scientific_name = 'Vireo olivaceus'
            output_species_labels.append(scientific_name)
            continue

    if len(df_sub) == 0:
        scientific_name = ebird_label
    else:
        scientific_name = df_sub['scientific name'].tolist()[0]
    output_species_labels.append(scientific_name)

# check if output labels exist in targets
target_labels = dl.ds.get_available_labels(ANNOTATION_COLUMN)

for species in target_labels:
    if not species in output_species_labels:
        print(species)
        breakpoint()

    
for i, x in enumerate(dl):
    dataloader = x['dataloader']
    selection_table = x['selection_table']
    print(f"inference for {i} out of {len(dl)}")

    preds = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batchpreds = model(batch).detach().numpy()
            preds.append(batchpreds)

    preds = np.concatenate(preds,axis=0)

            

# # audio = torch.tensor(np.random.normal(size=(1, 16000)))
# # with torch.no_grad():
# #     x = model(audio)
# #     print(x)
# #     print(x.size())

# # print(dl[3]['selection_table'])

# print(next(iter(dl[3]['dataloader'])))