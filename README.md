# End-to-End Neural Diarization

This repository is a modified version of the [Original Repo](https://github.com/BUTSpeechFIT/EEND.git).

## Installation

First, install the required libraries:

```bash
pip install -r requirements.txt
```

## Directory Structure

Before running the code, create the following directory structure:

```
DB/
├── real/
│   ├── train/
│   ├── dev/
│   ├── test/
└── simu/
    ├── train/
    ├── dev/
    └── test/
```

## Running the Code

To train EEND-EDA and perform speaker diarization, run:

```bash
chmod +x run.sh
./run.sh
```

This will start the training and inference process.