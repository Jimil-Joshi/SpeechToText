---
license: mit
tags:
- audio
- automatic-speech-recognition
datasets:
- librispeech_asr
---

# SpeechT5 (ASR task)

SpeechT5 model fine-tuned for automatic speech recognition (speech-to-text) on LibriSpeech.

This model was introduced in [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

SpeechT5 was first released in [this repository](https://github.com/microsoft/SpeechT5/), [original weights](https://huggingface.co/ajyy/SpeechT5/). The license used is [MIT](https://github.com/microsoft/SpeechT5/blob/main/LICENSE).

Disclaimer: The team releasing SpeechT5 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model Description

Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.

Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

## Intended Uses & Limitations

You can use this model for automatic speech recognition. See the [model hub](https://huggingface.co/models?search=speecht5) to look for fine-tuned versions on a task that interests you.

Currently, both the feature extractor and model support PyTorch.

## Citation

**BibTeX:**

```bibtex
@inproceedings{ao-etal-2022-speecht5,
    title = {{S}peech{T}5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing},
    author = {Ao, Junyi and Wang, Rui and Zhou, Long and Wang, Chengyi and Ren, Shuo and Wu, Yu and Liu, Shujie and Ko, Tom and Li, Qing and Zhang, Yu and Wei, Zhihua and Qian, Yao and Li, Jinyu and Wei, Furu},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    month = {May},
    year = {2022},
    pages={5723--5738},
}
```

## How to Get Started With the Model

Use the code below to convert a mono 16 kHz speech waveform to text.

```python
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
example_speech = dataset[0]["audio"]["array"]

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

predicted_ids = model.generate(**inputs, max_length=100)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
```
