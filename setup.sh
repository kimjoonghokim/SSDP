#!/bin/bash
pip install --upgrade pip setuptools wheel
pip uninstall torch torchaudio torchvision triton vllm pydantic flash-attn -y
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2
pip install -r requirements.txt