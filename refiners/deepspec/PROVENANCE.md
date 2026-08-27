# Vendored from deepseek-ai/DeepSpec

The files in this directory are copied from the official DeepSpec repository
(https://github.com/deepseek-ai/DeepSpec) so the Markov-head baseline runs the
AUTHORS' OWN implementation, unmodified except for import-path adaptation:

- `markov_head.py`  <- deepspec/modeling/dspark/markov_head.py  (VanillaMarkov)
- `sampling.py`     <- deepspec/utils/sampling.py

All credit for these files belongs to the DeepSpec authors. Please see their
repository for the license. Our loader (`refiners/markov.py`) only remaps our
checkpoint's `w1/w2` tensors onto `markov_w1/markov_w2` and drives the head's
native sequential decode.
