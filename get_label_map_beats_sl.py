from esp_data import dataset_from_config
from representation_learning.configs import RunConfig 
import json

# with open('configs/data_configs/aaai_train/data_animalspeak_nfs.yml', 'r') as f:
#    cfg = yaml.safe_load(f)


# cfg = DatasetConfig(**cfg)

run_cfg = RunConfig.from_sources(yaml_file='configs/run_configs/aaai_train/sl_beats_animalspeak.yml', cli_args=())


dataset, metadata = dataset_from_config(run_cfg.dataset_config.train_datasets[0])

label_map = metadata['label_from_feature']['label_map']

with open('sl_beats_animalspeak_label_map.json', 'w') as f:
    json.dump(label_map, f)