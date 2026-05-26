# TML2026 Task 2 - Model Stealing: Stolen Model Detection

This repository contains our submission for **TML Task 2: Stolen Model Detection**.

---

## Clone the Repository

``` bash
  git clone https://github.com/Meethiravi/tml26-task2.git
```

## Install the Dependencies

  ``` bash
pip install -r requirements.txt
```

## Prepare Model Files


All required models (target + suspect models) can be downloaded from:
``` bash
https://huggingface.co/SprintML/tml26_task2/tree/main

```

## Run the code
``` bash

condor_submit mods.sub
```
## Replace API Key

``` bash  
In submission.py replace "YOUR_API_KEY" with your actual API Key.
Also replace "PATH/TO/YOUR/SUBMISSION.csv" with your actual filepath.
```

## Submit the score
``` bash
condor_submit submit.sub
```
