from sklearn_crfsuite import CRF
class CRFModel(object):
    def __init__(self,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        features = [self.sentence_to_features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [self.sentence_to_features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists

    def word_to_features(self,sentence, i):
        """
        抽取单个字的特征
        :param sentence:
        :param i:
        :return:
        """
        word = sentence[i]
        prev_word = "<s>" if i == 0 else sentence[i - 1]
        next_word = "</s>" if i == (len(sentence) - 1) else sentence[i + 1]
        # 使用的特征：
        # 前一个词，当前词，后一个词，
        # 前一个词+当前词， 当前词+后一个词
        features = {
            'w': word,
            'w-1': prev_word,
            'w+1': next_word,
            'w-1:w': prev_word + word,
            'w:w+1': word + next_word,
            'bias': 1
        }
        return features

    def sentence_to_features(self,sentence):
        """
        抽取序列特征
        :param sentence:
        :return:
        """
        return [self.word_to_features(sentence, i) for i in range(len(sentence))]
