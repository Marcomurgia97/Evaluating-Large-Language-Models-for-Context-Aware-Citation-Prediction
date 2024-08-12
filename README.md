# Evaluating Large Language Models for Context-Aware Citation Prediction

This repo contains code for automating the citation prediction problem using transformers and related experiments for performances evaluation 
## Prerequisites

The experiments were carried out on windows 11, using python 3.9.13

To replicate the experiments it's compulsory to have an nvidia GPU supporting CUDA

Ensure you have an api key for mistral ("MISTRAL_API_KEY" in the environment variables) and a read access token ("READ_HF_TOKEN" env. variable) for Huggingface

## Installing
Steps are the following:

Clone the repo:
```
git clone https://github.com/Marcomurgia97/Evaluating-Large-Language-Models-for-Context-Aware-Citation-Prediction.git
```
Install transformers
```
pip install transformers==4.40.0
```
install pytorch with cuda
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Install mistralai
```
pip install mistralai
```
Install huggingface-hub
```
pip install huggingface-hub
```
Install peft
```
pip install peft
```
Install bitsandbytes
```
pip install bitsandbytes
```
Install accelerate
```
pip install accelerate
```
Install safetensors
```
pip install safetensors
```
## Running the tests
move to fst/snd/trd method and run (for example for the fst method, but is the same for all of the other methods)
```
python citation_suggestor_fstMethod.py --k <precision> --sentence <context> --entity <entity> --path_test_set <whereTestTestIsStored> --path_result <whereToSaveOutputs>
```
### Configuration Parameters

#### Precision

The precision parameter can be set to one of the following values:
- `1`
- `3`
- `5`

#### Sentence

The sentence parameter can be one of the following:
- `sentence_no_context`
- `sentenceWithContext`

#### Entity

The entity parameter can be one of the following:
- `titles_same_paper`
- `titles_other_papers`
- `abstract_same_paper`
- `abstract_other_papers`
- `abstract&title_same_paper`
- `abstract&title_other_papers`

To evalute the output of snd/trd method
```
python compute_acc.py --k <precision> --sentence <context> --entity <entity> --path <whereOutputIsStored>
```
## Datasets
You can find the CIPS dataset here: [CIPS](https://drive.google.com/file/d/1ZWv2K8fMZFWCTk8khVVkqCAJv6dbDkZS/view?usp=drive_link)

You can find the training set used for the fine tuning of Llama3 here: [CIPS-10k-Llama3](https://drive.google.com/file/d/13UPexMjm9H_NcaSjhoB8d6uYoBGmsNak/view?usp=sharing)

You can find the training set used for the fine tuning of Mistral here: [CIPS-10k-mistral](https://drive.google.com/file/d/1CTA8VVyKEpi_u9Tb7TfP3ImldVj6NWyL/view?usp=sharing)

You can find the test set here: [Test_set](https://drive.google.com/file/d/1UlifoUu1gnX9857UQCoS-dRGqcVd0ARh/view?usp=sharing)


## Fine tuning
Check the notebook unsloth_fineTuning.ipynb

## Outputs
You can find the responses used for the evalution of the models here: [Responses](https://drive.google.com/file/d/1HtbO1ucnvVVyeb4v1X0kqighMyPKd6Tn/view?usp=sharing)

## Models
You can find the fine tuned models (lora adapters) here: 

[LLama3](https://huggingface.co/MarcoMurgia97/Llama3-8B-FT)

[Mistral](https://huggingface.co/MarcoMurgia97/Mistral-7B-FT)

