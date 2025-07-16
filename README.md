# A repo for representation learning in bioacoustics

#  To run:
- `uv sync`
- uv run repr-learn train --config configs/run_configs/efficientnet_base.yml

#  To extend:
To add a model, subclass ModelBase in representation_learning/models/base_model.py and add it to get_model in
representation_learning/models/get_model.py.


# Repo TODO:
- Add augmentations
- Add preprocessors that help select audio windows (e.g. activity detection)
- Pick and add first SSL model
- Verify EfficientNet training run
- Add evaluation
- See if the tests run
- Test the wandb and mlflow backends
- Replace dummy data backend with the real one
