These fixtures pin the BEATs checkpoint linear-probe regression numbers.

Each file contains:
- initial_loss: cross-entropy on a fixed synthetic batch before training
- final_loss: cross-entropy after 10 SGD steps on the same fixed batch

The corresponding test is `tests/integration/test_beats_checkpoint_regression.py`.

If you intentionally change embedding extraction, checkpoint loading, or config
handling, regenerate these fixtures by running the helper script documented in
that test file (or by re-running the probe and updating the numbers).

