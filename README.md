## AN IMAGE IS WORTH 16x16 WORDS
*Envirment `pypoject.toml`,Model train on local and get 36% acc at val*

## Distributed Training
`m_train.py` can use `OMP_NUM_THREADS=1 torchrun --nproc_per_node=<NUM_OF_GPUs> m_train.py --other_args`

## Log
During the training, we obserived overfitting.
So add L1 regularization and

## Focal Loss

## CheckPoint



## Grad-CAM