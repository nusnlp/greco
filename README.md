# System Combination via Quality Estimation for Grammatical Error Correction
This repository provides the code to easily score, re-rank, and combine corrections from Grammatical Error Correction (GEC) models, as reported in this paper:
> System Combination via Quality Estimation for Grammatical Error Correction <br>
> [Muhammad Reza Qorib](https://mrqorib.github.io/) and [Hwee Tou Ng](https://www.comp.nus.edu.sg/~nght/) <br>
> The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) ([PDF](https://arxiv.org/abs/2310.14947))

## Installation
Please install the necessary libraries by running the following commands:
```
pip install -e requirements.txt
wget -P models https://sterling8.d2.comp.nus.edu.sg/~reza/GRECO/checkpoint.bin
wget https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz
tar -xf m2scorer.tar.gz
```
Please check whether the installed PyTorch matches your hardware CUDA version.

To also run other quality estimation models, please run the following commands:
```
git clone https://github.com/nusnlp/neuqe
git clone https://github.com/thunlp/VERNet
git clone https://github.com/kokeman/SOME
```
And download the model checkpoints from 
- https://github.com/nusnlp/neuqe to `checkpoints/neuqe` folder.
- https://github.com/nusnlp/neuqe to `checkpoints/vernet` folder.
- https://github.com/kokeman/SOME to `checkpoints/some` folder.

## Quality Estimation
### Scoring hypotheses in your code
You can import the GRECO class from `models.py`, instantiate the class, and pass the source(s) and hypotheses (in the form of python list of strings) to the `.score()` function.
```
import torch
from models import GRECO

model = GRECO('microsoft/deberta-v3-large').to(device)
model.load_state_dict(torch.load('models/checkpoint.bin))
model.score(source, hyphoteses)
```

### Correlation coefficient
Get the scores on all text by running this command. In this example, we will also score the text with SOME.
```
python score_all.py --auto --data_dir data/conll-official/texts --output_path outputs/greco_scores.json --model greco --lm_model microsoft/deberta-v3-large --checkpoint models/checkpoint.bin --source_file data/conll-source.txt --batch_size 16
python score_all.py --auto --data_dir data/conll-official/texts --output_path outputs/some_scores.json --model some --source_file data/conll-source.txt --batch_size 16
```
Get the gold F0.5 score for each sentence by running this command.
```
python m2_for_corr.py --data_dir data/conll-official/reports --scorer m2scorer --output_path outputs/target.json
```

Calculate the correlation by running this command
```
python correlation.py --system_A outputs/greco_scores.json --system_B outputs/some_scores.json --target outputs/target.json --metric spearman
```

## Re-ranking
### Reproducing re-ranking F0.5 score
Run the following to re-rank the corrections
```
python rerank.py --data_dir data/conll-official/texts --source_file data/conll-source.txt --auto --output_path outputs/greco_rerank.out --model greco --lm_model microsoft/deberta-v3-large --checkpoint models/checkpoint.bin --batch_size 16
```
Run the following to get the F0.5 score
```
python2 m2scorer/scripts/m2scorer.py outputs/greco_rerank.out data/conll-2014.m2
```

### Re-ranking your top-_k_ model outputs
You can run the same command as above but change the data path in the `--data_dir` argument. For all _k_, print the _k_-th best correction for each source sentence into a single file inside a folder, and pass that folder path to the `--data_dir` argument. The code will read all files inside that folder. You can check the `data/conll-official/texts` as an example.

## System Combination
### Reproducing system combination F0.5 score
Run the following command to reproduce the BEA-2019 test result
```
python run_combination.py --model greco --lm_model microsoft/deberta-v3-large --output_path outputs/bea-test.out --beam_size 16 --batch_size 16 --checkpoint models/checkpoint.bin --data data/test-m2/Riken-Tohoku.m2 data/test-m2/Kakao-Brain.m2 data/test-m2/UEDIN-MS.m2 data/test-m2/T5-Large.m2 data/test-m2/GECToR-XLNet.m2 data/test-m2/GECToR-Roberta.m2 --vote_coef 0.4 --edit_scores edit_scores/bea-test_score.json --score_ratio 0.7
```
Then, compress outputs/bea-test.out into a zip file and upload it to https://codalab.lisn.upsaclay.fr/competitions/4057#participate

Run the following command to reproduce the CoNLL-2014 test result
```
python run_combination.py --model greco --lm_model microsoft/deberta-v3-large --output_path outputs/conll-2014.out --beam_size 16 --batch_size 16 --checkpoint models/checkpoint.bin --data data/conll-m2/Riken-Tohoku.m2 data/conll-m2/UEDIN-MS.m2 data/conll-m2/T5-Large.m2 data/conll-m2/GECToR-XLNet.m2 data/conll-m2/GECToR-Roberta.m2 --vote_coef 0.4
```

Run the following to get the F0.5 score
```
python2 m2scorer/scripts/m2scorer.py outputs/conll-2014.out data/conll-2014.m2
```

## Retraining the model
Run the following command to train a new model
```
python train.py --do_train --model_name_or_path microsoft/deberta-v3-large --output_dir models/new_model --learning_rate 2e-5 --word_dropout 0.25 --save_strategy epoch --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 15 --alpha 1 --data data/train.json --data_mode hierarchical --edit_weight 2.0 --rank_multiplier 5
```

## License
The source code and models in this repository are licensed under the GNU General Public License Version 3 (see [License](./LICENSE.txt)). For commercial use of this code and models, separate commercial licensing is also available. Please contact Hwee Tou Ng (nght@comp.nus.edu.sg)