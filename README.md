# An example of a Python project for processing BCI data

I use this project to process BCI data from a CSV file and visualize it with matplotlib and mne.

## Setup Instructions

1. Create a virtual environment in the project folder:

```bash
 python -m venv venv 
```


2. Activate the virtual environment:

```bash
venv\Scripts\activate 
```


3. Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```


## Usage

To process a CSV file:

```bash
python analyze_eeg.py [input]
```
- where input is the full file path of your csv file generated from the Unicorn Recorder


## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt
