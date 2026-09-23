# HIATUS 

The goal of this project is to develop a pipeline for authorship obfuscation in English, Russian, Arabic, and Chinese (zh, zh-Hant) with fine-grained control over stylistic features. Throughout, we use the terms "style axis" and "author invariant" interchangeably to mean stylistic properties (e.g., use of sarcasm, word lengths, sentence lengths, frequency of function words) that are preserved within texts of the same author but vary between authors.

## Code

### HIATUS Main 

This project is meant to be run on an HPC cluster, mainly Hyak's Klone. Follow the [Hyak Klone instructions](https://hyak.uw.edu/docs/) to set up an account. Then, setup `miniconda` following [these instructions](https://hyak.uw.edu/docs/tools/python/#miniconda3). Lastly, follow instructions in [Disk Storage Management with Conda](https://hyak.uw.edu/blog/conda-disk-storage/) to properly set the paths used for storing environments and packages with conda. Also, make sure to set up the [ProxyJump alternative](https://hyak.uw.edu/docs/tools/vsc-proxy-jump) to use VSCode on a compute node.

Run the following commands on a compute node with at least one GPU (this is needed to verify the current CUDA version installed). First, create a `tmux` session typing `tmux`. Create an empty `conda` environment and activate it:
```
conda create -n hiatus --no-default-packages python=3.10.18
conda activate hiatus
```
Install the appropriate PyTorch version. First, verify the CUDA version with `nvidia-smi`. Then, search if compatible [wheels](https://download.pytorch.org/whl/torch/) exist (press `Cmd+F` and type the CUDA version e.g. 12.5). If not, install the latest compatible version. For example, for CUDA 12.5:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*Important:* this project was developed with `torch=2.7.0`, `torchaudio=2.7.0`, `torchvision=0.22.0`, and CUDA 12.x.
Install the base project dependencies. If the following step takes too long, you can safely detach from the session by pressing `Ctrl+B` then `Ctrl+D`. Follow the remaining steps only after the conclusion of the installation.
```
pip install -r requirements-main.txt
```
Install the local package in editable mode. 
```
pip install --no-build-isolation --no-deps --editable .
```
Download `spacy` language support for Chinese, Russian and English.
```
python -m spacy download en_core_web_lg
python -m spacy download ru_core_news_lg 
python -m spacy download zh_core_web_lg
```
Set up your Hugging Face Hub account.
```
hf auth login
```
Set your `HF_HOME` environment variable to a directory in `gscratch/stf/{user_name}` by adding a line like
```
export HF_HOME=/gscratch/stf/{user_name}/hugging_face
```
to your `.bashrc`. Similarly, add an environment variable to redirect download of HanLP models used in `langcheck`:
```
export HANLP_HOME=/gscratch/stf/mpotto/hanlp
```
Now, for running the Arabic models we need to set up `CoreNLP`. First, create a `CORE_NLP` environment variable to redirect the download of CoreNLP models, as in
```
export CORENLP_HOME=/gscratch/stf/mpotto/corenlp
```
Then, within a Python script or REPL:
```
import stanza
stanza.install_corenlp()
```
This will install CoreNLP in `CORENLP_HOME`. Now, install the most recent Arabic model. Within a script or REPL:
```
stanza.download_corenlp_models(model='arabic', version='4.5.10')
```

Finally, go to `src/hiatus/config.py` and set the variables with your personal information. 

**Important**: do not make any part of this repository public.

### HIATUS Eval

Metrics for evaluation in HIATUS are defined by an external technical team in partnership with IARPA/HIATUS. Because of this, their codebase is not necessarily compatible with our main HIATUS codebase (whose dependencies are listed in the `requirements-main.txt` and managed through `conda`). In this step, we describe the steps needed to set up the evaluation environment for HIATUS.

Create a new `conda` environment.
```
conda create -n hiatus-eval --no-default-packages python=3.11
conda activate hiatus-eval
```
Check the CUDA version and install the appropriate wheel.
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
Now, Install `hiatus-metrics` requirements for `klone`. First, go to `requirements.txt` and remove the `torch` dependency from the list of requirements. Then,
```
pip install path/to/hiatus-metrics
```
Then, do the following:
```
pip install transformers==4.49.0 einops peft bitsandbytes
```
This step is necessary to ensure that the attribution model works. Now, go to the root of the `hiatus-metrics` and do the following:
```
pip install --no-build-isolation --no-deps --editable .
```
Finally, create symbolic links from the `src/hiatus/config.py` and `src/hiatus/data.py` in our repo to `hiatus/config.py` and `hiatus/hrs.py` in the `hiatus-metrics` repo.
```
cd path/to/hiatus-metrics/hiatus
ln -sf path/to/hiatus/src/hiatus/data.py hrs.py
ln -sf path/to/hiatus/src/hiatus/config.py config.py
```
Check if the symlinks resolve to the right paths. Within `hiatus-metrics/hiatus`, type:
```
ls -l 
```
And check if the arrow `->` points to the right path.

**Printing Equal Error Rate**: since `hiatus-eval` is installed in development mode, we can patch their `src` package to print the Equal Error Rate for the original and privatized texts. For this, go to `hiatus-metrics/hiatus/privacy/experiment.py`, and add the following lines of code to line 340 indented to match the `if` statement.
```python
summary[f"{attribution_metric_name} (original)"] = out_context_attribution_score
summary[f"{attribution_metric_name} (privatized)"] = out_context_deattribution_score
```

## Naming Conventions

All Slurm files should be named following the following standard:
```
{pipeline_step}_{pipeline-suffix, tbt/tfs}_{language acronym}{version}.slurm
```
For instance, for the "obfuscation" step with PromptRemix in a model trained from scratch (i.e., creating classifiers for that language based on externally sourced data) for Chinese in its base version:
```
obfuscation_promptremix-tfs_zh0.slurm
```

## Internal Folder structure

Below we detail the folder structure of this project and the data repositories. `slurm` scripts are meant to be run on `hyak` when we are confident that scaling-up will improve results. For compiled, end-to-end versions of the steps enumerated in the scripts, use `slurm` scripts in the `scripts/e2e` directory. 

* Source code (in `\mmsf1\home`)
```text
.           
├── prompts/                    # Prompts for Style/PromptRemix paraphrasing.
│   └── prompts_{lang}      
├── scripts/ 
│   └── 1-data                  # Download data for training classifiers/adapters.
│   ├── 2-paraphrase
        └── slurm               # Scripts to run on `hyak` GPU nodes. "large" version.
│   ├── .
│   └── utils
├── src/  
│   └── hiatus
├── pyproject.toml              # Basic information for building the source package in src/hiatus.
├── README.md               
├── requirements-main.txt       # Core dependencies of HIATUS main environment.
├── environment-main.yml        # Rich description of HIATUS main environment.
├── hiatus.mplstyle             # Matplotlib style for this project.
└── TUTORIAL.md                 # Tutorial with `hyak` and Style/PromptRemix guides.
```

* Individual data (in `\gscratch\stf\{user_name}`).
```text
.
├── classifiers                 # Save classifiers for Style/PromptRemix.
│   └── {lang}
│       └── {style}_classifier
├── adapter                     # Save adapters for StyleRemix.
│   └── {lang}
│       └── {style}_adapter
├── data
│   └── {lang}
│       ├── raw                 # Raw data used as basis for paraphrasing.
│       ├── paraphrases
│       │   ├── original        # Paraphrased data (as output by paraphraser LLM).
│       │   └── processed       # Paraphrased data (after quality checks).
│       └── evaluation
├── logs                        # Logs for Slurm files executed on `hyak`.
└── privatized                  # Obfuscated (privatized) texts.
```
* HIATUS HRS data and byproducts in `/ifml1/hiatus`
```
.
├── hrs                         # HRS validation data.
├── metrics                     # Save metrics as .json
│   └── {lang}
├── plots                       # Save plots as .pdf
│   └── {lang}
├── tables                      # Save tables as .csv
│   └── {lang}
```
