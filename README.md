## AN IMAGE IS WORTH 16x16 WORDS

*Environment `pypoject.toml`,Model train on local and get 36% acc at val*

### Distributed Training
`m_train.py` can use `OMP_NUM_THREADS=1 torchrun --nproc_per_node=<NUM_OF_GPUs> m_train.py --other_args`
it premise your machine has 2more GPUs.

### Log
During the training, we observed overfitting.
`Epoch [60/60] | Train Loss: 0.6494 | Train Acc: 0.5387 | Val Loss: 1.1327 | Val Acc: 0.3248Lr: 1.00e-07`
can approximately see Train acc 64% but Val acc 32%.

So add L1 regularization

### Focal Loss

### Check Point



### Grad-CAM