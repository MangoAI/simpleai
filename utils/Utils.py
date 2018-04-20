from datetime import datetime

def parseDateTime(str):
    if ':' in str:
        return datetime.strptime(str, "%Y-%m-%d_%H:%M:%S")
    return datetime.strptime(str, "%Y%m%d_%H%M%S")