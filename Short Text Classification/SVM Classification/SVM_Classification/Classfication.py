from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
def vocab_convert_to_vector(file_name,class_num,max_length):
    model = Word2Vec(LineSentence(file_name),
                     size=5,
                     window=3,
                     min_count=1  # 词出现的个数小于1则不进入训练
                     )
    model.save("w2c_model.model")
    matrix = []
    sentences = []
    labels=[]
    temp=[]
    h=0
    for line in open(file_name, encoding="utf-8"):
        words = list(line.strip().split(" "))
        for i in words:
            matrix=matrix+list(model.wv[str(i)])
        if len(matrix)>max_length:
            matrix=matrix[:max_length]
        else:
            size=(max_length-len(matrix))
            for i in range(size):
                temp.append(0)
            matrix=matrix+temp
            temp=[]
        sentences.append(matrix)
        labels.append(class_num)
        matrix = []
    return sentences,labels


def predict(train_data,train_label,test_data):

    pred=OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_data, training_labels).predict(test_data)
    #clf = svm.SVC(decision_function_shape='ovo')
    #clf.fit(train_data, train_label)
    #pred=clf.predict(test_data)
    return pred

def tp_calcu(predictions,actuals,value):
    count=0
    size=len(predictions)
    for i in range(size):
        if predictions[i]==actuals[i]==value:
            count=count+1
    return count
def count_values(input_list,input_value):
    count=0
    size=len(input_list)
    for i in range(size):
        if input_list[i]==input_value:
            count=count+1
    return count

def metrics(predictions,actuals):
    tp_zeros=tp_calcu(predictions,actuals,0)
    tp_ones = tp_calcu(predictions, actuals, 1)
    tp_twos = tp_calcu(predictions, actuals, 2)

    actual_zeros=count_values(actuals,0)
    actual_ones = count_values(actuals, 1)
    actual_twos = count_values(actuals, 2)

    predict_zeros=count_values(predictions,0)
    predict_ones = count_values(predictions, 1)
    predict_twos = count_values(predictions, 2)

    try:
        precision_zeros = float(tp_zeros) / (float(predict_zeros))
    except ZeroDivisionError:
        precision_zeros = 0
    try:
        precision_ones = float(tp_ones) / (float(predict_ones))
    except ZeroDivisionError:
        precision_ones = 0
    try:
        precision_twos = float(tp_twos) / (float(predict_twos))
    except ZeroDivisionError:
        precision_twos = 0

    try:
        recall_zeros = float(tp_zeros) / (float(actual_zeros))
    except ZeroDivisionError:
        recall_zeros = 0
    try:
        recall_ones = float(tp_ones) / (float(actual_ones))
    except ZeroDivisionError:
        recall_ones = 0
    try:
        recall_twos = float(tp_twos) / (float(actual_twos))
    except ZeroDivisionError:
        recall_twos = 0

    try:
        f1_zeros = (2 * (precision_zeros * recall_zeros)) / (precision_zeros + recall_zeros)
    except ZeroDivisionError:
        f1_zeros = 0
    try:
        f1_ones = (2 * (precision_ones * recall_ones)) / (precision_ones + recall_ones)
    except ZeroDivisionError:
        f1_ones = 0
    try:
        f1_twos = (2 * (precision_twos * recall_twos)) / (precision_twos + recall_twos)
    except ZeroDivisionError:
        f1_twos = 0

    print("第一类的precision,recall,f1分别为",precision_zeros,recall_zeros,f1_zeros)
    print("第二类的precision,recall,f1分别为",precision_ones,recall_ones,f1_ones)
    print("第三类的precision,recall,f1分别为",precision_twos,recall_twos,f1_twos)


if __name__ == '__main__':
    cooking_data_train,cooking_labels_train=vocab_convert_to_vector("cooking.txt",0,55)
    music_data_train, music_labels_train = vocab_convert_to_vector("music.txt", 1,55)
    video_data_train, video_labels_train = vocab_convert_to_vector("video.txt", 2,55)
    training_data=cooking_data_train+music_data_train+video_data_train
    training_labels=cooking_labels_train+music_labels_train+video_labels_train
    cooking_data_test, cooking_labels_test = vocab_convert_to_vector("cooking_test.txt",0,55)
    music_data_test, music_labels_test = vocab_convert_to_vector("music_test.txt", 1,55)
    video_data_test, video_labels_test = vocab_convert_to_vector("video_test.txt", 2,55)
    test_data = cooking_data_test + music_data_test + video_data_test
    test_labels = cooking_labels_test + music_labels_test + video_labels_test
    predictions=list(predict(training_data,training_labels,test_data))
    print(predictions)
    metrics(predictions,test_labels)
