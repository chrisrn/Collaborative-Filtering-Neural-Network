## Collaborative filtering

### Setup environment
We download and unzip the csv files for [Book-Crossing dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) 
into `data` folder. Then we create a python 2.7 virtual environment and we activate it:
```
virtualenv -p python venv
source venv/bin/activate
```
Now we are ready to install the packages we need:
```
pip install -r requirements.txt
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.14.0-cp27-none-linux_x86_64.whl
```

### Training process
Before running the training script, we can modify the parameters from `params.json` file.
If you run this project for the first time you should set the below flags to True, because they create help files. Otherwise, you set them to False and we use the already created files.
* `clean_titles`: If this flag is True, we remove the symbol ';' from title column of BX-Books.csv file and a new BX-Books-new.csv file is created for the books data.
* `multiple_isbn_pickle`: If this flag is True, a big dictionary is extracted into a pickle file for the creation of unique isbns.
* `create_data`: If this flag is True, the final dataset for training after the processing is extracted into
data folder with name `dataset.csv`. In this way, we do not need to process the data every time we train.

When the above flags are True, the whole data processing lasts 10 minutes. But after the first
run, when we set these flags to False, the data processing is really fast! Let's view
the parameters that we can modify:
```
{"data_dir": "data",
 "debug_data": false,
 "debug_rows": 50000,
 "clean_titles": true,
 "multiple_isbn_pickle": true,
 "create_data": true,
 "test_split": 0.2,
 "validation_split": 0.1,
 "batch_size": 256,
 "epochs":  15,
 "learning_rate":  0.01,
 "adaptive_learning_rate": true,
 "adaptive_lr_patience_epochs": 1,
 "adaptive_lr_decay": 0.2,
 "min_adaptive_lr": 0.000001,
 "exponential_lr": false,
 "num_epochs_per_decay": 2,
 "lr_decay_factor": 0.2,
 "early_stopping": false,
 "early_stopping_min_change": 0.05,
 "early_stopping_patience_epochs": 3,
 "fine-tuning-file": "",
 "save_model": false,
 "epochs_per_save": 2,
 "model_dir": "model",
 "gpus": 0,
 "log_test_predictions": true,
 "forward_pass": false
 }
```

`data_dir` and `model_dir` are the folders of data and models that are saved. It's good to
use the folders 'data' and 'model' into this directory for flexibility. If you have gpus and you
want to accelerate training, you can give the number of gpus to `gpus` parameter. But this
code is not tested, because we used only CPU for training. If we turn `forward_pass` to true we should
provide a `fine-tuning-file` for restoring the weights and just predict without training.
Now we are ready to train!
```
python train_eval.py
```