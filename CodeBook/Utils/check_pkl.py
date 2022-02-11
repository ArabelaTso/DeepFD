import pickle


def show_pkl_file(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    print(data)
    return data


if __name__ == '__main__':
    filepath = "../RQ2/IMDB/config_random_b807bb19-3a91-41a3-aa3a-504abacbb497/config_Adagrad_772151cd-143d-3c8c-b61d-d7146baada40.pkl"
    config = show_pkl_file(filepath)
