# A repo for representation learning in bioacoustics

#  🗣️ To run:
- uv sync
- uv run python representation_learning/run_train.py --config configs/run_configs/efficientnet_base.yml


# Repo TODO:
- Add augmentations
- Add preprocessors that help select audio windows (e.g. activity detection)
- Pick and add first SSL model
- Verify EfficientNetB0 training run
- Add evaluation
- See if the tests run
- Test the wandb and mlflow backends
- Replace dummy data backend with the real one
