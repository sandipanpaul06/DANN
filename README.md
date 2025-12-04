Repository for the domain-adaptive neural network approach discussed in DeGiorgio, M., Arnab, S.P., Fumagalli, M., *"AI solutions for evolutionary genomics of nonmodel species"*.

# Image dataset generation
Image dataset generation follows Arnab et al. 2025, *"Efficient detection and characterization of targets of natural selection using transfer learning"*. GitHub repo: [https://github.com/sandipanpaul06/TrIdent](https://github.com/sandipanpaul06/TrIdent).

# Training DANN

**Arguments**

--src_sweep

Path to the source-domain dataset containing sweep images stored in a NumPy .npy file.

--src_neut

Path to the source-domain dataset containing neutral images stored in a NumPy .npy file.

--tgt_sweep

Path to the target-domain dataset containing sweep images stored in a NumPy .npy file.

--tgt_neut

Path to the target-domain dataset containing neutral images stored in a NumPy .npy file.

--src_train

Number of source-domain sweep samples and source-domain neutral samples used for training; the value applies per class.

--src_val

Number of source-domain sweep samples and source-domain neutral samples used for validation; the value applies per class.

--tgt_train

Number of target-domain sweep samples and target-domain neutral samples included in the unlabeled training set; the value applies per class.

--tgt_val

Number of target-domain sweep samples and target-domain neutral samples included in the unlabeled validation set; the value applies per class.

--epochs

Total number of training epochs for the DANN model.

--batch

Batch size used during training.

**Example run**

```sh
python DANN.py \
    --src_sweep ./path/to/src_sweep.npy \
    --src_neut  ./path/to/src_neutral.npy \
    --tgt_sweep ./path/to/tgt_sweep.npy \
    --tgt_neut  ./path/to/tgt_neutral.npy \
    --src_train 5000 \
    --src_val 1000 \
    --tgt_train 5000 \
    --tgt_val 1000 \
    --epochs 50 \
    --batch 32
```
