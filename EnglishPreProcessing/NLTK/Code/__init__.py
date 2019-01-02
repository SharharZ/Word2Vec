from EnPreProcess import EnPreProcess

if __name__ == '__main__':
    data_path = "data/"
    output_path = "output/"
    epp = EnPreProcess(data_path, output_path)
    epp.en_pre_main()
