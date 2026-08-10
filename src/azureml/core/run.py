"""No-op replacement for azureml.core.run.Run.

The original ePillID code logs metrics/tags/images to an Azure ML run context.
Locally we just print scalar logs and write images to ./azureml_logs/ so the
training code runs unmodified.
"""
import os
import json


class _LocalRun:
    def __init__(self):
        self._tags = {}
        self._logdir = "azureml_logs"
        os.makedirs(self._logdir, exist_ok=True)
        self._metrics_path = os.path.join(self._logdir, "metrics.jsonl")

    @staticmethod
    def get_context():
        return _LocalRun()

    def tag(self, key, value=None):
        self._tags[key] = value

    def log(self, name, value, **kwargs):
        # keep console output light; persist everything to a jsonl file
        try:
            with open(self._metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": name, "value": str(value)}) + "\n")
        except Exception:
            pass

    def log_list(self, name, value, **kwargs):
        self.log(name, value)

    def log_image(self, name, plot=None, path=None, **kwargs):
        try:
            if plot is not None:
                out = os.path.join(self._logdir, f"{name}.png".replace(":", "-"))
                plot.savefig(out)
        except Exception:
            pass

    def upload_file(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass


# original code calls: from azureml.core.run import Run; run = Run.get_context()
Run = _LocalRun
