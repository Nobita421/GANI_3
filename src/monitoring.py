import json
import os
from datetime import datetime

class Monitor:
    def __init__(self, log_file="usage_logs.json"):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)

    def log_request(self, crop, disease, count):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "crop": crop,
            "disease": disease,
            "count": count
        }
        
        # Simple/Naive logging (concurrency might be an issue in prod, acceptable for demo)
        try:
            with open(self.log_file, 'r+') as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Monitoring Log Error: {e}")

monitor = Monitor()
