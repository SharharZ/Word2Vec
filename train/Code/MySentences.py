import os


class MySentences:
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        for root, dirs, files in os.walk(self.data_path):
            for eachfiles in files:
                data_path = os.path.join(root, eachfiles).replace("\\", "/")
                for line in open(data_path):
                    if line != '\n':
                        yield line.split()
        print("This is a memory-friendly iterator")
