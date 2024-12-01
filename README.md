# Implementation-of-Neural-Machine-Translation-of-Rare-Words-with-Subword-Units-on-Azerbaijani
This project is designed to apply Neural Machine Translation techniques proposed in https://arxiv.org/abs/1508.07909 on Azerbaijani language for NSP task

How to use the code base:
- Install https://huggingface.co/datasets/allmalab/AZE-NSP dataset into /data_source/ directory
- Run huggingface_tokenizers.ipynb in order to create a tokenizer on Azerbaijani language
- Run huffman_trees.ipynb in order to create a huffman tree that can be used by decoder
- Follow instruction below after modifying "/config" directory with the preferred settings

How to install libraries:
- for extra libraries: python -m pip install --no-cache-dir <package_name>
- to initiate the virtual environment, run: pipenv install -r requirements.txt && pipenv shell
- to run the experiments, run: bash experiment.sh
- to add configurations for your own experiments, modify .yaml files under /config/ path