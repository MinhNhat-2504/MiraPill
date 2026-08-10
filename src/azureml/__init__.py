# Local no-op shim of the azureml SDK so the original ePillID training code
# (train_cv.py / train_nocv.py / multihead_trainer.py) runs locally without
# Azure ML. Only the small API surface actually used is provided.
