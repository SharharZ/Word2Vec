
import os

if __name__ == '__main__':

    dataPath = "corpus/"
    outputPath = "model/"
    for root, dirs, files in os.walk(dataPath):
        for eachfiles in files:
            dataPath = os.path.join(root, eachfiles).replace("\\", "/")
            print(dataPath)
