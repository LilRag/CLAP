# CLAP
<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/logo.PNG" alt="The Contrastive Language-Audio Pretraining Model Architecture" width="60%"/>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2211.06687"><img src="https://img.shields.io/badge/arXiv-2211.06687-brightgreen.svg?style=flat-square"/></a>
  <a href="https://pypi.org/project/laion-clap"><img src="https://badge.fury.io/py/laion-clap.svg"/></a>
  <a href="https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/clap"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue"/></a>
</p>
 
### This forked repository uses Contrastive Language-Audio Pretraining (CLAP) to test for similarity between the generated audio and text prompt. 

With CLAP, you can extract a latent representation of any given audio and text for your own model, or for different downstream tasks.

All codes are comming officially with the following paper, accepted by IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2023:
 - [Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687)


## Architecture
Contrastive Language-Audio Pretraining, known as CLAP. Referring to the CLIP (Contrastive Language-Image Pretraining) architecture, the CLAP architecture is as follows.  
<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/audioclip-arch.png" alt="The Contrastive Language-Audio Pretraining Model Architecture" width="60%"/>
</p>


## Audio Genre Classification and Similarity Measurement

This project utilizes the [CLAP model](https://huggingface.co/laion/clap-htsat-fused) for audio and text embedding extraction, allowing users to measure the similarity of text prompts with audio files. It enhances genre classification accuracy and supports broader audio-to-text matching capabilities.

## Acknowledgments
- This project integrates the [CLAP model](https://huggingface.co/laion/clap-htsat-fused) by LAION for generating audio embeddings and improving genre classification accuracy.
