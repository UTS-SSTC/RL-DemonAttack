import csv, os


class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.w.writeheader()
        self.f.flush()

    def log(self, **kwargs):
        self.w.writerow(kwargs)
        self.f.flush()

    def close(self):
        self.f.close()
