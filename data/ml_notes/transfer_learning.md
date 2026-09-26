# Transfer Learning

Transfer learning reuses a model trained on a large, general dataset as the
starting point for a new, usually smaller, task. Instead of learning features
from scratch, you inherit representations that already capture useful structure.

## Two common strategies

- **Feature extraction.** Freeze the pretrained backbone and use it only to turn
  inputs into fixed embedding vectors, then train a small classifier head on
  those embeddings. This is fast, works on a CPU, and needs little data — a good
  fit when the new task is similar to the original.
- **Fine-tuning.** Unfreeze some or all of the backbone and continue training it
  on the new task with a small learning rate. This adapts the representations to
  the new domain and usually scores higher, at the cost of more compute and a
  higher risk of overfitting on small datasets.

## When it helps

Transfer learning shines when labelled data is scarce. A frozen ImageNet
ResNet-18 plus a small head can reach strong accuracy on a few thousand images,
and a pretrained transformer fine-tuned for a few epochs typically beats a
from-scratch model on text classification. The larger and more general the
pretraining, the more the new task benefits.
