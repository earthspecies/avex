from representation_learning.models.perch import PerchModel
import torch
import numpy as np
import pandas as pd
from zeroshot_detection_eval.data.dataloader import DataloaderBuilder
from zeroshot_detection_eval.eval.evaluation import Scorer
from esp_data.io import filesystem
import os 
from tqdm import tqdm

DATASET = "wabad"
BATCH_SIZE = 4
WINDOW_SIZE = 0.5
N_WORKERS = 8
ANNOTATION_COLUMN = "Species"

out_fp = f"perch_{DATASET}_{WINDOW_SIZE}_{ANNOTATION_COLUMN}.yaml"

model = PerchModel(0)
model.eval()
dl = DataloaderBuilder(DATASET, WINDOW_SIZE, WINDOW_SIZE, BATCH_SIZE, N_WORKERS, sr=32000)

# convert output labels ebird -> scientific name
output_labels = pd.read_csv('perch_labels.csv')['ebird2021_code'].tolist()

if DATASET == "powdermill":
    info_fp = "Clements-v2024-October-2024-rev.csv"
    if not os.path.exists(info_fp):
        fs = filesystem("gcs")
        gcp_path = "gs://esp-ml-datasets/wabad/v0.1.0/Clements-v2024-October-2024-rev.csv"
        fs.get(gcp_path, info_fp)

    info_df = pd.read_csv(info_fp)
    info_df = info_df[~pd.isna(info_df["scientific name"])]
if DATASET == "wabad":
    info_fp = "eBird_Taxonomy_v2021.csv"
    if not os.path.exists(info_fp):
        fs = filesystem("gcs")
        gcp_path = "gs://esp-ml-datasets/wabad/v0.1.0/eBird_Taxonomy_v2021.csv"
        fs.get(gcp_path, info_fp)

    info_df = pd.read_csv(info_fp)
    info_df = info_df[~pd.isna(info_df["SCI_NAME"])]
    info_df['species_code'] = info_df["SPECIES_CODE"]
    info_df['scientific name'] = info_df["SCI_NAME"]


output_species_labels = []
print("correcting species labels")
for ebird_label in tqdm(output_labels):
    df_sub = info_df[info_df['species_code'] == ebird_label]
    if DATASET == "powdermill":
        if ebird_label == "haiwoo":
            scientific_name = 'Leuconotopicus villosus'
            output_species_labels.append(scientific_name)
            continue

        if ebird_label == "reevir":
            scientific_name = 'Vireo olivaceus'
            output_species_labels.append(scientific_name)
            continue

    if DATASET == "wabad":
        correction_dict = {}
        # correction_dict = {"bkhbat1" : "Batis minor",
        #                     "subwar1" : "Curruca cantillans",
        #                     "leswhi1" : "Curruca curruca",
        #                     "subwar1" : "Curruca iberiae",
        #                     "grnjay1" : "Cyanocorax yncas",
        #                     "ruwant4" : "Herpsilochmus frater",
        #                     "blewhe1" : "Oenanthe hispanica",
        #                     "blcapa2" : "Oreolais rufogularis",
        #                     "grnwoo1" : "Picus viridis",
        #                     "gowbar1" : "Psilopogon chrysopogon",
        #                     "reevir" : "Vireo olivaceus",
        #                     "butwoo2" : "Xiphorhynchus guttatus",}

        if ebird_label in correction_dict.keys():
            scientific_name = correction_dict[ebird_label]
            output_species_labels.append(scientific_name)
            continue

    if len(df_sub) == 0:
        scientific_name = ebird_label
    else:
        scientific_name = df_sub['scientific name'].tolist()[0]
    output_species_labels.append(scientific_name)

if DATASET == "wabad" : 
    bonus_columns = [("bkhbat1", "Batis minor"),
                     ("subwar1" , "Curruca cantillans"),
                     ("leswhi1" , "Curruca curruca"),
                     ("subwar1" , "Curruca iberiae"),
                     ("grnjay1" , "Cyanocorax yncas"),
                     ("ruwant4" , "Herpsilochmus frater"),
                     ("blewhe1" , "Oenanthe hispanica"),
                     ("blcapa2" , "Oreolais rufogularis"),
                     ("grnwoo1" , "Picus viridis"),
                     ("gowbar1" , "Psilopogon chrysopogon"),
                     ("reevir" , "Vireo olivaceus"),
                     ("butwoo2" , "Xiphorhynchus guttatus")]
else:
    bonus_columns = []

# check if output labels exist in targets
target_labels = dl.ds.get_available_labels(ANNOTATION_COLUMN)

for species in target_labels:
    if not species in output_species_labels:
        if not species in [x[1] for x in bonus_columns]:
            print(species)

S = Scorer(temp_dir=f"{DATASET}_{WINDOW_SIZE}")

for i, x in enumerate(dl):
    dataloader = x['dataloader']
    selection_table = x['selection_table']
    print(f"inference for {i} out of {len(dl)}")
    if len(selection_table) == 0:
        print(f"skipping example {i} because it has no ground-truth events; see github issue for sed_scores_eval")
        continue

    preds = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batchpreds = model(batch).detach().numpy()
            preds.append(batchpreds)

    preds = np.concatenate(preds,axis=0)
    preds_df = pd.DataFrame(preds, columns=output_species_labels)
    if DATASET == "wabad":
        bonus_preds_df = pd.DataFrame(preds, columns=output_labels)
        bonus_preds_df = bonus_preds_df[[x[0] for x in bonus_columns]]
        bonus_preds_df.columns = [x[1] for x in bonus_columns]
        preds_df = pd.concat([preds_df, bonus_preds_df], axis =1)
    preds_df = preds_df[target_labels]
    preds = preds_df.to_numpy()
    preds = 1. / (1. + np.exp(-preds))

    S.update(preds, target_labels, 1/WINDOW_SIZE, selection_table, ANNOTATION_COLUMN)
    # if i>0:
    #     break

S.compute_scores(output_fp = out_fp, delete_temp=False, num_jobs=12)
