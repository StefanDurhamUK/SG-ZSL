# The code is for the paper of 'Sentinel-Guided Zero-Shot Learning: A Collaborative Paradigm without Real Data Exposure' and 'Privacy-Enhanced Zero-Shot Learning via Data-Free Knowledge Transfer'.

## Environment

- python:3.6.13
- pytorch:1.7.1

## Dataset

- The entire dataset of this project is available in Teams group channels. (Folder name:DatasetForGroupProject)
- [Dataset link](https://teams.microsoft.com/_#/school/files/Group%20Study?threadId=19%3A5c11800f52bf4af1b5bb33d792679435%40thread.tacv2&ctx=channel&context=DatasetForGroupPorject&rootfolder=%252Fteams%252FPerceptionLab%252FShared%2520Documents%252FGroup%2520Study%252FDatasetForGroupPorject)


## Setting Explanation

- 1. GZSL
        - TRAIN:
            - T: allclasses real data (train split)
            - G: generate allclasses fake data
            - S: train on allclasses fake data
        - TEST:
            - real data allclasses (test split) -> S -> class space: allclasses
            - obtain acc_unseen, acc_seen, H

-
    2. ZSL
        - TRAIN:
            - T: allclasses real data (train split)
            - G: generate allclasses fake data
            - S: train on allclasses fake data
        - TEST:
            - real data unseenclasses (test split) -> S -> class space: allclasses (more challenging)
            - obtain acc_unseen

-
    3. AZSL extension

    - TRAIN:
        - T: seenclasses real data (trainval split)
        - G: generate seenclasses fake data
        - S: train on seenclasses fake data
    - TEST:
        - a. ZSL
            - unseenclass bert-> G -> unseenclasses fake data
            - unseenclasses fake data -> Z_net training (unseenclasses space)
            - real unseen data -> Z_net -> obtain acc_unseen
        - b. GZSL
            - unseenclass bert-> G -> unseenclasses fake data
            - unseen fake data + real trainval data -> Z_net training (allclasses space)
            - real unseen + test_seen data-> Z_net -> obtain acc_unseen,acc_seen, H
    

