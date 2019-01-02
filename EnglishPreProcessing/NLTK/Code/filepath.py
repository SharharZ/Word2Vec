import os
import re

from  nltk.corpus import  stopwords

def WriteResult(datapath, resultPath):
    txtpath = os.path.basename(datapath)
    cwd = os.getcwd()
    print(cwd)
    path = re.sub(str(datapath).split('/')[-1], '', datapath)
    path = re.sub(str(path).split('/')[0], outputPath, path)
    dirPath = cwd + '/' + re.sub('//', '/', path)

    print(dirPath)
    mkdir(dirPath)
    # # self.mkdir(str(resultPath).replace(str(resultPath).split('/')[-1],''))
    # f = open(Path + 'Civil Engineering e-Books_Lemmatize.txt', 'a', encoding='utf8')  # 将结果保存到另一个文档中
    # for i in result:
    #     if i != '\n':
    #         f.write(str(i) + ' ')
    #     else:
    #         f.write(str(i))
    # # f.write(str(result))
    # f.close()

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    # path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在    True
    # 不存在  False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path + '  create successfully!')
        print('\n')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path+'  The directory exists!!')
        print('\n')
        return False

if __name__ == '__main__':
    stop_words = stopwords.words('english')
    print(stop_words)
    print(type(stop_words))
    stop_words.append('cid')
    print(stop_words)
