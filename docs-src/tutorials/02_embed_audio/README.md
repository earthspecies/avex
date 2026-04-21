# Embed Audio

Generic batch embedding extraction script. Processes any directory of `.wav` or `.flac` files and saves per-file embeddings using both avex models.

**Inspired by:** [Perch / embed_audio.ipynb](https://github.com/google-research/perch/blob/main/embed_audio.ipynb)

## Usage

```bash
python embed_audio.py --config config.yaml
```

Edit `config.yaml` to point `audio.input_dir` at your data. Embeddings are saved to `output.embeddings_dir/<model_name>/<filename>.pt`.

## Output format

Each `.pt` file contains a `(n_windows, embedding_dim)` tensor — one mean-pooled embedding per audio window. Window length and hop size are set in `config.yaml`.

## Notes

- Already-extracted files are skipped (idempotent).
- Embeddings from this script are used as input to examples 06 and 07.
