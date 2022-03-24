import pickle
def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model
def sort_by_lengths(word_lists, tag_lists):
    """
    将句子按长度排序
    :param word_lists:
    :param tag_lists:
    :return:
    """
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices