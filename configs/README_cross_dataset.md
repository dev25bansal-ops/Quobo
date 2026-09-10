"""OPEN-10: cross-dataset replication harness.

Same 7-arm pipeline on a second dataset (TrashNet by default) to pre-empt
the single-dataset rejection. TrashNet (Stanford, MIT license) ships as
class folders of jpgs, so unlike TACO it needs no cropping — only a
resize-free pass through the same embedding/selection/classifier stack.

Setup (one-time): download the TrashNet zip from the repo README link and
extract so that data/trashnet/<class>/*.jpg exists. Then:

    .venv/Scripts/python.exe -m src.quobo.run_experiment configs/experiment_trashnet.yaml

The experiment code is dataset-agnostic: it only reads crops folders. This
config points the features tag at TrashNet's class list.
"""
