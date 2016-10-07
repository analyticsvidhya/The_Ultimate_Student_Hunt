Input data:
Input files are referred to as 'train.csv' and 'test.csv' in current directory.
This is read in lines 25-26 in xgb.py, lines 34-34 in krs.py & lines 22-23 in keras_epoch_blender.py.


Requirements (on top of python builtin packages):
Python 3.5.2 (Anaconda preferred)
XGBoost 0.6 (Built from source, commit 4733357)
Keras 1.0.8 (Built from source, commit f0d9867)
Theano 0.8.2
Pandas 0.18.1
Numpy 1.11.1

Run order:
python xgb.py
python krs.py
python keras_epoch_blender.py
python ensemble.py

Final submission is called "final prediction" (no file extension as per requested)
