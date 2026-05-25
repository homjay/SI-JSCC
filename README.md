# CBJSCC

This repository is the public code entry for **Channel-Blind Joint Source-Channel
Coding for Wireless Image Transmission**.

- [Inference branch](https://github.com/homjay/SI-JSCC/tree/inference): original inference notebook and released pretrained weights.
- [Training branch](https://github.com/homjay/SI-JSCC/tree/training): cleaned training and evaluation code for reproducing the paper experiments.

The training branch contains the reorganized CBJSCC implementation and also includes the ADJSCC and DeepJSCC baseline modules used for comparison.

## Author's Note

Thanks to the recent development of AI, I can now organize and release some of
my previous code much faster.

CBJSCC was completed during 2022-2023. The original goal was to show that a
Deep JSCC model can still obtain strong channel-adaptive ability without
depending on SNR. Later, I found that many works use SNR as an explicit
condition for source-channel coding, while this work was asking whether the
model itself can adapt to the channel without that condition.

Some communication experts pointed out that this may lack a mathematical proof
(although this sounds very reasonable, it is really hard to prove). So I tried
to modify the modules, including combinations of convolution and self-attention,
to raise the performance ceiling while keeping the model adaptive to channel
changes. (In hindsight, I should have used a more carefully designed backbone
at that time.)

Overall, for this paper, I think the main contribution is to show that Deep
JSCC models can adapt to channel variation. The module improvement is only a
small part of the work. I think this point is important because it can help us
improve model transmission under real wireless channels in later work.

After this work, I spent a lot of effort trying to understand why this happens,
and also tried some more practical systems. These follow-up works will be published soon.
