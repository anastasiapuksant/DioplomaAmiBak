# -*-coding:utf-8-*-
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import random
import nltk
import math
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn import svm
import numpy as np
import pymorphy2
from pystruct.models import ChainCRF, GridCRF
from pystruct.learners import FrankWolfeSSVM
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold

from transliterate import translit

morph = pymorphy2.MorphAnalyzer()

lePOS = LabelEncoder()
lePOS.fit(["POST", "NOUN","ADJF","ADJS","COMP","VERB","INFN","PRTF","PRTS","GRND","NUMR","ADVB","NPRO","PRED","PREP","CONJ","PRCL","INTJ","None"])
leANim = LabelEncoder()
leANim.fit(["anim","inan","None"])
leGEnd = LabelEncoder()
leGEnd.fit (["masc","femn","neut","None"])
leNMbr = LabelEncoder()
leNMbr.fit(["sing", "plur","None"])
leCAse = LabelEncoder()
leCAse.fit(["nomn", "gent", "datv", "accs", "ablt", "loct", "voct",	"gen1", "gen2", "acc2", "loc1", "loc2","None"])
leASpc = LabelEncoder()
leASpc.fit(["perf", "impf","None"])
leTRns = LabelEncoder()
leTRns.fit(["tran","intr","None"])
lePErs = LabelEncoder()
lePErs.fit(["1per","2per","3per","None"])
leTEns = LabelEncoder()
leTEns.fit(["pres","past","futr", "None"])
leMOod = LabelEncoder()
leMOod.fit(["indc","impr","None"])
leINvi = LabelEncoder()
leINvi.fit(["incl","excl","None"])
leVOic = LabelEncoder()
leVOic.fit(["actv","pssv","None"])
lePNkt = LabelEncoder()
lePNkt.fit = ["True" , "False"]

leShape = LabelEncoder()
#leShape.fit = ["xx" , "Xxx", 'xXxx','XX','XX-00','00']

OUT = 0
UNARY = 1
BEGIN = 2
INNER = 3
LAST = 4


def walk(dir):

    paths_tok =[]
    paths_obj = []
    paths_span = []
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            if path[-8:-1] == ".object":
                paths_obj.append(path)
            elif path[-7:-1] == ".token":
                paths_tok.append(path)
            elif path[-6:-1] == ".span":
                paths_span.append(path)
        else:
            walk(path)
    return [paths_span,paths_tok,paths_obj]

# def getShape(txt):
#     if txt == '':
#         return '-'
#     UP = 'X'
#     LOW = 'x'
#     DIG = '0'
#     PUNKT = '-'
#     s = ''
#     f1 = False
#     old = '1'
#     for i in range(0, len(txt)):
#         if txt[i].isupper():
#             tmp = UP
#         elif txt[i].isdigit():
#             tmp = DIG
#         elif txt[i] in ['-','<','"','>','«','»','—','.']:
#             tmp = PUNKT
#         else:
#             tmp = LOW
#         if old != tmp and not f1:
#             s += tmp
#
#         if old == tmp:
#             f1 = True
#         else:
#             f1 = False
#         old = tmp
#     return s

class ExtractFeatures:
    def __init__(self):
        pass

    def is_none(self, x):
        if x is None:
            y = "None"
        else:
            y = str(x)
        return y

    def is_true(self, x):
        if x is True:
            return 1
        else:
            return 0

    def info(self, txt, size_token):
        token = np.zeros(( size_token))
        t = txt.lower()
        info = morph.parse(t)[0]#берем только первый разбор
        p = info.tag
        token[0] = 10*lePOS.transform(self.is_none(p.POS)) # Part of Speech, часть речи
        token[1] = 10*leANim.transform(self.is_none(p.animacy)) # одушевленность
        token[2] = leGEnd.transform(self.is_none(p.gender))        # род (мужской, женский, средний)
        #token[3] = leNMbr.transform(self.is_none(p.number))  # число (единственное, множественное)
        #token[4] = leCAse.transform(self.is_none(p.case))  #падеж
        #token[5] = leASpc.transform(self.is_none(p.aspect))        # вид: совершенный или несовершенный
        #token[6] = leTRns.transform(self.is_none(p.transitivity))  # переходность (переходный, непереходный)
        #token[7] = lePErs.transform(self.is_none(p.person))        # лицо (1, 2, 3)
        #token[8] = leTEns.transform(self.is_none(p.tense))         # время (настоящее, прошедшее, будущее)
        #token[9] = leMOod.transform(self.is_none(p.mood))          # наклонение (повелительное, изъявительное)
        #token[10] = leINvi.transform(self.is_none(p.involvement))   # включенность говорящего в действие
        #token[11] = leVOic.transform(self.is_none(p.voice))         # залог (действительный, страдательный)

        token[3] = (5*int('Geox' in p)) # локация?
        token[4] =(5*int('Name' in p)) # имя?
        token[5] =(5*int('Surn' in p)) # фамилия?
        token[6] =(5*int('Patr' in p)) # отчество?
        token[7] =(int('Init' in p)) # инициал?
        token[8] =(5*int('Sgtm' in p)) # sgtm?????
        token[9] =(int('Orgn' in p)) # организация?
        token[10] =(int('Fixd' in p)) # неизменяемое?


        token[12] = self.is_true(txt.isupper()) #все ли буквы заглавные
        token[13] = 4*self.is_true(txt.istitle()) #первая буква - заглавная
        token[13] = self.is_true(txt.isnumeric()) #только цифры unicode
        token[14] = 10*self.is_true(txt.isalpha()) #все буквы и без пробелов
        token[15] = self.is_true(txt.isalnum()) #все буквы или числа
        token[16] = self.is_true(txt.islower()) #все буквы в нижнем регистре

        return token

    def fit(self, documents, y = None):

        return self.fit_transform(documents)

    def fit_transform(self, documents, y = None):
        n = len(documents)

        size_token = 17 #отведенный размер для одного токена
        k = 8 #кол-во токенов в одном объекте

        matrix = np.zeros((n, size_token * k))
        count1 = 0
        for i in documents:
            count2 = 0
            #print("count1 = ",count1)
            # if count1 == 2211:
            #     print (i)
            for j in i:
                matrix[count1, (count2*size_token) :((count2+1)*size_token)] = self.info(j,size_token)
                count2 += 1
                # print("count2 = ", count2)
            count1 += 1
        # print("matrix debug--------start--------------------")
        # print(matrix)
        # print("matrix debug--------end----------------------")
        return matrix

    def transform(self, X):
        """Applies transforms to the data, and the transform method of the
        final estimator. Valid only if the final estimator implements
        transform."""
        return self.fit_transform(X)

def inLocation(txt):
    lib = [
        'район',
        'деревня',
        'город',
        'село',
        'столица',
        'страна',
        'государство',
        'площадь',
        'улица',
        'река',
        'озеро',
        'поселок',
        'колхоз',
        'область',
        'округ',
        'северный',
        'южный',
        'восточный',
        'западный',
        'остров',
        'станция',
        'планета',
        'спутник',
        'космос',
        'бульвар',
        'долина'
    ]
    if txt in lib:
        return 1
    else:
        return 0
def inPerson(txt):
    lib = [
        'президент',
        'глава',
        'спортсмен',
        'знаменитость']
    if txt in lib:
        return 1
    else:
        return 0
def inOrg(txt):
    lib = [
        'имени',
        'музей',
        'станция',
        'учреждение',
        'фонд',
        'предприятие',
        'частный',
        'федерация',
        'международный',
        'центр',
        'компания',
        'проект',
        'организация',
        'корпорация',
        'комитет',
        'союз',
        'агенство',
        'отдел',
        'театр',
        'отдел'
    ]
    if txt in lib:
        return 1
    else:
        return 0

class ExtractFeaturesToArray:
    def __init__(self):
        pass

    def is_none(self, x):
        if x is None:
            y = "None"
        else:
            y = str(x)
        return y

    def is_true(self, x):
        if x is True:
            return 1
        else:
            return 0

    def getContextPunkt(self,txt):
        t = txt.lower()
        info = morph.parse(t)[0]#берем только первый разбор
        p = info.tag
        normForm = morph.normal_forms(t)

        res = np.zeros(12)
        res[0] = (int('CONJ' in p)) #союз?
        res[1] = (int('PRCL' in p)) #частица?
        res[3] = (int('PREP' in p)) #предлог?
        res[4] = (int(txt == '"')) #предлог?
        res[5] = (int(txt == '«')) #предлог?
        res[6] = (int(txt == '»')) #предлог?
        res[7] = (int(txt == '(')) #предлог?
        res[8] = (int(txt == ')')) #предлог?
        res[9] = (int(txt == '<')) #предлог?
        res[10] = (int(txt == '>')) #предлог?
        res[11] = (int(txt in ['.', ',', '...', ':', ';', '!', '?'])) #предлог?
        return res
    def getShape(self,txt):
        tmp = np.zeros(7)
        for i in range (0, min(7, len(txt))):
            c = txt[i]
            if c.isupper():
                tmp[i]=(ord('X'))
            elif c.islower():
                tmp[i]=(ord('x'))
            elif c.isdigit():
                tmp[i]=(ord('D'))
            elif c == '-':
                tmp[i]=(ord('-'))
        return tmp
    def getMorph(self,txt):
        t = txt.lower()
        info = morph.parse(t)[0]#берем только первый разбор
        p = info.tag

        if 'LATN' in p:
            t = translit(txt,'ru')
            info = morph.parse(t)[0]#берем только первый разбор
            p = info.tag
        res = np.zeros(16)
        tmp_res = []

        #tmp_res.append(lePOS.transform(self.is_none(p.POS))) # Part of Speech, часть речи

        tmp_res.append(int('ADJF' in p)) #прилаательное?
        tmp_res.append(int('ADJS' in p)) #краткое прилагательное?
        tmp_res.append(int('COMP' in p)) #компаратив?

        tmp_res.append(int('INFN' in p)) #инфинитив?
        tmp_res.append(int('PRTF' in p)) #причатсие?
        tmp_res.append(int('PRTS' in p)) #краткое причатсие?
        tmp_res.append(int('GRND' in p)) #деепричастие?
        tmp_res.append(int('NUMR' in p)) #числительное?
        tmp_res.append(int('ADVB' in p)) #наречие?
        tmp_res.append(int('NPRO' in p)) #местоимение?
        tmp_res.append(int('PRED' in p)) #предикат?
        tmp_res.append(int('PREP' in p)) #предлог?
        tmp_res.append(int('CONJ' in p)) #союз?
        tmp_res.append(int('PRCL' in p)) #частица?
        tmp_res.append(int('INTJ' in p)) #междометие?
        #tmp_res.append(int('INTJ' in p) + int('PRCL' in p)+ int('CONJ' in p)
        #              +int('PREP' in p)+int('PRED' in p)+int('NPRO' in p)
        #               +int('ADVB' in p))

        tmp_res.append(int('PRTF' in p)+ int('PRTS' in p)
                       +int('GRND' in p)+3*int('INFN' in p))

        tmp_res.append(int('NOUN' in p)) #существительное?
        tmp_res.append(leCAse.transform(self.is_none(p.case)))  #падеж
        tmp_res.append(int('anim' in p)) # одушевленность
        tmp_res.append(int('masc' in p)) #мужской?
        tmp_res.append(int('femn' in p)) #женский?
        tmp_res.append(int('neut' in p)) #средний
        tmp_res.append(int('sing' in p))  # число (единственное, множественное)
        # tmp_res.append(5*int('NOUN' in p)+2*int('masc' in p)+2*int('femn' in p)
        #                +int('neut' in p)+int('sing' in p) + 5*(leCAse.transform(self.is_none(p.case))))


        tmp_res.append(int('VERB' in p)) #глагол?
        tmp_res.append(leASpc.transform(self.is_none(p.aspect)))        # вид: совершенный или несовершенный
        tmp_res.append(leTRns.transform(self.is_none(p.transitivity)))  # переходность (переходный, непереходный)
        tmp_res.append(lePErs.transform(self.is_none(p.person)))        # лицо (1, 2, 3)
        tmp_res.append(leTEns.transform(self.is_none(p.tense)))         # время (настоящее, прошедшее, будущее)
        tmp_res.append(leMOod.transform(self.is_none(p.mood)))          # наклонение (повелительное, изъявительное)
        tmp_res.append(leINvi.transform(self.is_none(p.involvement)))   # включенность говорящего в действие
        tmp_res.append(leVOic.transform(self.is_none(p.voice)))         # залог (действительный, страдательный)
        # tmp_res.append(10*int('VERB' in p) + leASpc.transform(self.is_none(p.aspect))
        #                +leTRns.transform(self.is_none(p.transitivity))+lePErs.transform(self.is_none(p.person))+
        #                leTEns.transform(self.is_none(p.tense))+leMOod.transform(self.is_none(p.mood))+
        #                leINvi.transform(self.is_none(p.involvement))+leVOic.transform(self.is_none(p.voice)))


        tmp_res.append(int('Geox' in p)) # локация?

        tmp_res.append(int('Name' in p)) # имя?
        tmp_res.append(int('Surn' in p)) # фамилия?
        tmp_res.append(int('Patr' in p)) # отчество?
        #tmp_res.append(10*int('Name' in p) + 10*int('Surn' in p)+ 10*int('Patr' in p))


        tmp_res.append(int('Init' in p)) # инициал?
        tmp_res.append(int('Sgtm' in p)) # sgtm?????

        tmp_res.append(int('Orgn' in p)) # организация?
        tmp_res.append(int('Fixd' in p)) # неизменяемое?
        tmp_res.append(int('Abbr' in p))
        tmp_res.append(int('Trad' in p))
        #tmp_res.append(7*int('Orgn' in p)+int('Fixd' in p)+int('Abbr' in p)
         #              +int('Sgtm' in p)+self.is_true(txt.isupper())+int('Trad' in p))




        tmp_res.append(self.is_true(txt.isupper())) #все ли буквы заглавные
        #tmp_res.append(self.is_true( txt.istitle()   )) #первая буква - заглавная

        tmp_res.append(self.is_true(txt.isnumeric())) #только цифры unicode
        tmp_res.append(self.is_true(txt.isalpha())) #все буквы и без пробелов
        tmp_res.append(self.is_true(txt.isalnum())) #все буквы или числа
        tmp_res.append(self.is_true(txt.islower())) #все буквы в нижнем регистре

        # tmp_res.append(self.is_true(txt.islower()) + self.is_true(txt.isalnum())+self.is_true(txt.isalpha())
        #                -2*self.is_true(txt.isnumeric())+5*self.is_true(txt.isupper()))

        res = np.array(tmp_res)
        return res
    def nearWords(self,txt):
        t = txt.lower()
        info = morph.parse(t)[0]#берем только первый разбор
        p = info.tag
        normForm = morph.normal_forms(t)

        res = np.zeros(3)
        res[0] = int(inLocation(normForm)) # характеристика локации
        res[1] = int(inPerson(normForm)) # характеристика персоны
        res[2] = int(inOrg(normForm)) # характеристика организации
        return res
    def getPosition(self, tokens):
        tmp = []
        if tokens[0] == '':
            tmp.append(0)
        elif tokens[4] == '':
            tmp.append(2)
        else:
            tmp.append(1)
        if(tmp[0] == 0 and self.is_true( tokens[2].istitle())): #первая буква - заглавная)
            tmp.append(10)
        else:
            tmp.append(0)
        res = np.array(tmp)
        return res
    def info(self,tokens):
        #tokens -2 -1 current 1 2
        features = np.zeros(0)

        #morphems
        features = np.concatenate((features,self.getMorph(tokens[1]),self.getMorph(tokens[2]),self.getMorph(tokens[3])))

        features = np.concatenate((features,self.getContextPunkt(tokens[1]),self.getContextPunkt(tokens[3])))

        features = np.concatenate((features,self.getPosition(tokens)))
        #features = np.concatenate((features,self.getShape(tokens[2])))
        norm = np.linalg.norm(features)
        return features


    def fit(self, documents, y=None):

        return self.fit_transform(documents, y)

    def fit_transform(self, documents, y = None):
        x = documents
        matrixOfSamples = []
        #answer = []
        #extrf = ExtractFeaturesToArray()
        count1 = 0
        len_features = len(self.info(['','','','','']))
        #print(x[0])

        for sentense in x:
            len_sample = len(sentense)
            sample = np.zeros((len_sample, len_features))
            window = ["",""]
            if len_sample > 0:
                window.append(sentense[0])
            else:
                window.append("")
            if len_sample > 1:
                window.append(sentense[1])
            else:
                window.append("")
            if len_sample > 2:
                window.append(sentense[2])
            else:
                window.append("")
            count2 = 0
            for index in range(0, len_sample):
                window.pop(0)
                if index+2<len_sample:
                    window.append(sentense[index+2])
                else:
                    window.append("")
                features = self.info(window)
                # cur= self.info(sentense[index],len_features)
                # if index != 0:
                #     prev= self.info(sentense[index-1],len_features)
                # else:
                #     prev = np.zeros(len_features)
                #     cur[22] = 1
                # if(index>1):
                #     prev2 = self.info(sentense[index-2],len_features)
                # else:
                #     prev2 = np.zeros(len_features)
                # if index < len_sample-1:
                #     next= self.info(sentense[index+1],len_features)
                # else:
                #     next = np.zeros(len_features)
                # if index < len_sample-2:
                #     next2= self.info(sentense[index+2],len_features)
                # else:
                #     next2 = np.zeros(len_features)
                # cur[21] = index
                # prev[21] = index-1
                # next[21] = index+1
               #koef = np.sqrt(np.sum((np.concatenate((prev2, prev, cur, next, next2)) * np.concatenate((prev2, prev, cur, next, next2)))))
                sample[count2, :] = features
                count2 += 1
            #print(sample.shape)
            matrixOfSamples.append(sample)
            #print(matrixOfSamples[0].shape)
            count1 += 1
        #answer = []

        #answer = np.array(answer)
        matrix = np.array(matrixOfSamples)
        return matrix


    def transform(self, X):
        """Applies transforms to the data, and the transform method of the
        final estimator. Valid only if the final estimator implements
        transform."""
        return self.fit_transform(X)



class NamedEntityRecognition:
    def __init__(self):
        self.classifierMNB = Pipeline([  #Multinomial Naive Bayes
                ('extract', ExtractFeatures()),
                #('encoding', MultiColumnLabelEncoder()),
                ('clf', MultinomialNB(alpha=0.5))
                ])
        # self.classifierMaxEnt = Pipeline([
        #         ('extract', ExtractFeatures()),
        #         #('encoding', MultiColumnLabelEncoder()),
        #         ('clf', nltk.maxent.MaxentClassifier.train(x, algorithm = 'gis', trace = 0, max_iter = 10))
        #         ])
        self.classifierMaxEnt_LogReg = Pipeline([ #Maximum Entropy
                ('extract', ExtractFeatures()),
                ('clf', linear_model.LogisticRegression())
                ])
        self.classifierCRF = Pipeline([ #CRF
                ('extract', ExtractFeaturesToArray()),
                ('clf', FrankWolfeSSVM(model=ChainCRF(), C=2, max_iter=10, tol=0.01))
                ])
        self.classifierSVM = Pipeline([ #Support Vector Machine
                ('extract', ExtractFeatures()),
                ('clf',  svm.LinearSVC())
                ])


        pass
    def train_N0CRF(self, x, y):

        # #работает!!начало---------------------------------
        # x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.7, random_state=0)
        #
        # self.classifierMNB.fit(x_train,y_train)
        # # x_test = [[u"пришел",u"человек"], ["Красная", "площадь"],["Федеральное","авиационное","управление"],["Вашингтоном"]]
        # # print(x_test)
        # y_result_MNB = self.classifierMNB.predict(x_test)
        # class_names = ['Org', 'Per','Loc','No Named Entity']
        # print('Multinomial Naive Bayes')
        # print(classification_report(y_test, y_result_MNB,target_names = class_names ))
        #
        # # print("Answer:", y_test)
        #
        #
        # #
        # # x_ME = ExtractFeatures()
        # # x_ME.fit_transform(x)
        # # MEclassif = nltk.maxent.MaxentClassifier.train(x_ME, algorithm = 'gis', trace = 0, max_iter = 10)
        # # y_test = self.classifierMaxEnt.predict(x_test)
        # # print("Answer1:", y_test)
        #
        #
        # self.classifierMaxEnt_LogReg.fit(x_train,y_train)
        # y_result_ME = self.classifierMaxEnt_LogReg.predict(x_test)
        # print('Maximum Entropy')
        # print(classification_report(y_test, y_result_ME ,target_names = class_names ))
        #
        # # print("AnswerMAXent:", y_test)
        #
        # # self.classifierCRF.fit(x,y)
        # # y_test = self.classifierCRF.predict(x_test)
        # # print("AnswerCRF:", y_test)
        # #работает!!конец---------------------------------
        #print("x = ",x)
        class_names = ['Org', 'Per','Loc','No Named Entity']
        kf = KFold(len(x), n_folds=3, shuffle=True)
        for train_index, test_index in kf:
            x_train = []
            x_test = []
            y_train = []
            y_test = []
            for i in train_index:
                x_train.append(x[i])
                y_train.append(y[i])
            for i in test_index:
                x_test.append(x[i])
                y_test.append(y[i])
            # x_train, x_test = x[train_index], x[test_index]
            # y_train, y_test = y[train_index], y[test_index]
            # print("x_test",x_test)


            # self.classifierMNB.fit(x_train,y_train)
            # y_result_MNB = self.classifierMNB.predict(x_test)
            # print('Multinomial Naive Bayes')
            # print(classification_report(y_test, y_result_MNB,target_names = class_names ))
            #
            # self.classifierMaxEnt_LogReg.fit(x_train,y_train)
            # y_result_ME = self.classifierMaxEnt_LogReg.predict(x_test)
            # print('Maximum Entropy')
            # print(classification_report(y_test, y_result_ME ,target_names = class_names ))

            self.classifierSVM.fit(x_train,y_train)
            y_result_SVM = self.classifierSVM.predict(x_test)
            print('Support Vector Machine')
            tmp1111 = classification_report(y_test, y_result_SVM ,target_names = class_names )#это строка
            #tmp = tmp1111.find("Loc",0,-1)

            #print(tmp)




            print("-----------------------------------------------------------------")
        pass
# х - последовательность предложений, а предложение - это последовательность токенов, у - последовательность меток по каждому предложение, размеченное предложение
    #  def train_CRF(self, x, y):
    #     class_names = ['Org', 'Per','Loc','No Named Entity']
    #     #x, y = np.array(x), np.array(y)
    #     kf = KFold(len(x), n_folds=3, shuffle=False)
    #     # for train_index, test_index in kf:
    #     #
    #     #     x_train, x_test = x[train_index], x[test_index]
    #     #     y_train, y_test = y[train_index], y[test_index]
    #     #     # for i in range (0,100):
    #     #     #     print (x_train[i], y_train[i])
    #     #     # print("_____________________________________")
    #     for train_index, test_index in kf:
    #         x_train = []
    #         x_test = []
    #         y_train = []
    #         y_test = []
    #         for i in train_index:
    #             x_train.append(x[i])
    #             y_train.append(y[i])
    #         for i in test_index:
    #             x_test.append(x[i])
    #             y_test.append(y[i])
    #
    #
    #         self.classifierCRF.fit(x_train,y_train)
    #         y_result_CRF = self.classifierCRF.predict(x_test)
    #         print('Conditional Random Field')
    #         print(classification_report(y_test, y_result_CRF ,target_names = class_names ))
    #
    #     pass

    # x = set of sentenses
    # y разметка каждого предложения
    def convertYtoCRF(self, y):
        answer = []
        for row in y:
            column = np.array(row).T
            #print(column)
            answer.append(column)
        return np.array(answer)
    def train_CRF(self, x, y):
        # class_names = ['No Named Entity', 'NONE', 'Org', 'Per','Loc',]
        # countSamples = len(x)
        #
        # matrixOfSamples = []
        # answer = []
        # extrf = ExtractFeatures()
        # count1 = 0
        # len_features = 17
        # print(x[0])
        # for sentense in x:
        #     len_sample = len(sentense)
        #     sample = np.zeros((len_sample, len_features))
        #
        #     count2 = 0
        #     for token in sentense:
        #         sample[count2, :] = extrf.info(token,len_features)
        #         count2 += 1
        #     #print(sample.shape)
        #     matrixOfSamples.append(sample.astype(int))
        #     #print(matrixOfSamples[0].shape)
        #     count1 += 1
        # answer = []
        # for row in y:
        #     column = np.array(row).T
        #     #print(column)
        #     answer.append(column)
        # answer = np.array(answer)
        # matrix = np.array(matrixOfSamples)

        # print("matrix debug--------start--------------------")
        # print(matrix)
        # print("matrix debug--------end----------------------"

        kf = KFold(len(x), n_folds=3, shuffle=True)
        for train_index, test_index in kf:
            x_train = []
            x_test = []
            y_train = []
            y_test = []
            for i in train_index:
                x_train.append(x[i])
                y_train.append(y[i])
            for i in test_index:
                x_test.append(x[i])
                y_test.append(y[i])
            #print(y_train[0])
            # for i in range (0,100):
            #      print ("x =", x_train[i], "y =", y_train[i])
            print("_____________________________________")


            self.classifierCRF.fit(x_train,self.convertYtoCRF(y_train))
            y_result_CRF = self.classifierCRF.predict(x_test)



            match = 0
            total = 0

            for i in range(0, len(y_result_CRF)):
                state = 0
                dif = 0
                print(x_test[i])
                print(np.array(y_test[i]))
                print(y_result_CRF[i])

                print('=====================================================')
                for j in range(0, len(y_result_CRF[i])):
                    test = y_test[i][j]
                    res = y_result_CRF[i][j]


                    if(state == 0):
                        if test != 0 and res == test:
                            state = 1
                            dif = 0
                        else:
                            state = 2
                    elif state == 1:
                        if(test == 3 ):
                            total += 1
                            state = 0
                            if(dif == 0):
                                match +=1
                        else:
                            dif += abs(res-test)
                    else:
                        if(test == 3):
                            total+=1
                            state = 0
            x_spans = getSpans(x_test, y_result_CRF)
            print('==========================getting spans=========================')
            print(x_spans)
            print('==========================end getting spans=========================')

            y_classes = self.classifierSVM.predict(x_spans)
            print("==========================predictions classes===========================")
            labels = ['N', 'N', 'Organization', 'Person', 'Location', 'Other']
            #le = preprocessing.LabelEncoder()
            #le.fit(y_classes)
            #print(le.classes_)
            for index in range(0, len(x_spans)):
                print(x_spans[index], labels[y_classes[index]+2])
            #print(le.classes_)
            print("==========================end predictions classes===========================")
            print(match, total)
            print('Conditional Random Field')
            str = [['Владимир','Владимирович','поехал','в','Таганрог','.'],
                   ['Владимир','котик111','Федоров','посетил','музей','имени','Ленина','.']]
            res = self.classifierCRF.predict(str)
            print(res)
#            print(classification_report(y_test, y_result_CRF ,target_names = class_names ))

        pass
namedTags = ['Name','Surn','Patr','Geox','Orgn']
def isNamed(txt):
    tags = morph.parse(txt)[0].tag
    score = 0
    for t in namedTags:
        if t in tags:
            score+=10
    if score > 9:
        return True
    else:
        return False

def staticAnalyze(sentense, mark):
    for i in range(0, len(sentense)):
        if(mark[i] == UNARY):
            if isNamed(sentense[i]):
                mark[i] = OUT
    for i in range(0, len(sentense)):
        if mark[i] == BEGIN:
            t = morph.parse(sentense[i])[0].tag
            if 'anim' in t and not isNamed(sentense[i]):
                if mark[i+1] != LAST:
                    mark[i+1] = BEGIN
                else:
                    mark[i+1] = UNARY
                mark[i] = OUT
    for i in range(0, len(sentense)):
        if(mark[i] == UNARY):
            t = morph.parse(sentense[i])[0].tag
            if 'Name' in t:
                if i > 0 and mark[i-1] == OUT:
                    t2 = morph.parse(sentense[i-1])[0].tag
                else:
                    t2 = []
                if 'Surn' in t2:
                    mark[i-1] = BEGIN
                    mark[i] = LAST
            elif 'Surn' in t:
                if i < len(sentense)-1 and mark[i+1]==OUT:
                    t2 = morph.parse(sentense[i+1])[0].tag
                else:
                    t2 = []
                if 'Name' in t2:
                    mark[i] = BEGIN
                    mark[i+1] = LAST


    return mark
def getSpans(sentenses, marks):

    result = []
    for index in range(0, len(sentenses)):
        spans = []
        span = []
        #marks2 = staticAnalyze(sentenses[index], marks[index])
        marks2 = marks[index]
        for markPos in range(0, len(marks[index])):
            #mark = marks[index][markPos]
            #use analyzer
            mark = marks2[markPos]

            token = sentenses[index][markPos]

            if(mark == UNARY):
                span.append(token)
                spans.append(span)
                span = []
            elif mark == BEGIN:
                span.append(token)
            elif mark == INNER:
                span.append(token)
            elif mark == LAST:
                span.append(token)
                spans.append(span)
                span = []
            elif mark == OUT:
                if(len(span) > 0):
                    spans.append(span)
                    span = []
        result.append(spans)




    return result

def getTokens(path):
    paths = (walk(path))

    spans = dict()
    for path_i in paths[0]:
        # #файл span-------------------------------------------------------------------------
        f = open(path_i,'r')

        span = dict()
        #classes = dict()
        #all_classes = set()
        for line in f:
            arr = line.split(" ")
            span.clear()

            span['type'] = arr[1]
            span['tokens_num'] = arr[5]

            l = list()
            i = 6

            while arr[i] != '#':
                i += 1

            for j in range (i+1, int(arr[5])+i+1,1):
                l.append(int(arr[j]))


            span['tokens'] = l

            spans[arr[0]] = span.copy()
        f.close()

    #файл obj--------------------------------------------
    objects = []
    for path_i in paths[2]:
        f = open(path_i,'r')


        for line in f:
            arr = line.split(" ")
            object = []
            object.append(int(arr[0]))#obj_id
            if arr[1] == 'LocOrg':
                object.append(1)
            elif arr[1] == 'Org':
                object.append(2)
            elif arr[1] == 'Person':
                object.append(3)
            elif arr[1] == 'Location':
                object.append(4)
            else:
                object.append(0)
            i = 2
            count = 0
            while arr[i] != '#':
                object.append(int(arr[i]))
                count += 1
                i += 1

            object.append(count)
            objects.append(object)
        f.close()

    arrayTokens = []
    tokens = dict() #ключ - id, значение - структура d
    for path_i in paths[1]:
        #файл tok----------------------------------------------------------
        f = open(path_i,'r')
        sentense = [];
        d = dict()
        for line in f:

            arr = line.split(" ")
            if arr[0] != '\n':
                d['sym'] = int(arr[1])#позиция символа в предложении, с которого начинается токен
                d['num'] = int(arr[2])#длина текстового поля
                if arr[3][-1] == "\n":
                    arr[3] = arr[3][0:-1]
                d['txt'] = arr[3]
                d['class'] = 0
                d['obj_id'] = 0
                d['tokens_num'] = 0 #сколько токенов в данной именной сущности

                sentense.append(int(arr[0]))
                tokens[int(arr[0])] = d.copy()
            else:
                arrayTokens.append(sentense.copy())
                sentense.clear()



        f.close()

    result_x = []
    result_y = []
    result_CRF_y = []
    max_object_size = 0
    for obj in objects:
        num = int(obj[-1])
        if num > max_object_size:
            max_object_size = num

        for spanId in range(2, num + 2, 1):
            #print ('object ',i)
            #print (j)
            #print('id_span ',i[j])
            #print(spans[str(i[j])])
            for tokenId in (spans[str(obj[spanId])]['tokens']):
                #if tokens[k]['class'] != 1 and tokens[k]['tokens_num'] < : # организация превыше локации
                # разметка последовательности токенов

               if tokens[tokenId]['class'] < 2 or tokens[tokenId]['tokens_num'] < num :

                    tokens[tokenId]['class'] = obj[1]
                    tokens[tokenId]['obj_id'] = obj[0]
                    tokens[tokenId]['tokens_num'] = num
    resSentense = list()
    for sentense in arrayTokens:
        resSent = [tokens[token] for token in sentense]
        resSentense.append(resSent)
    return resSentense

def retriveNamedEntity(path):
    paths = (walk(path))

    spans = dict()
    for path_i in paths[0]:
        # #файл span-------------------------------------------------------------------------
        f = open(path_i,'r')
        span = dict()
        #classes = dict()
        #all_classes = set()
        for line in f:
            arr = line.split(" ")
            span.clear()

            span['type'] = arr[1]
            span['tokens_num'] = arr[5]

            l = list()
            i = 6

            while arr[i] != '#':
                i += 1

            for j in range (i+1, int(arr[5])+i+1,1):
                l.append(int(arr[j]))


            span['tokens'] = l

            spans[arr[0]] = span.copy()
        f.close()

    #файл obj--------------------------------------------
    objects = []
    for path_i in paths[2]:
        f = open(path_i,'r')
        for line in f:
            arr = line.split(" ")
            object = []
            object.append(int(arr[0]))#obj_id
            if arr[1] == 'LocOrg':
                object.append(0)
            elif arr[1] == 'Org':
                object.append(2)
            elif arr[1] == 'Person':
                object.append(3)
            elif arr[1] == 'Location':
                object.append(4)
            else:
                object.append(0)
            i = 2
            count = 0
            while arr[i] != '#':
                object.append(int(arr[i]))
                count += 1
                i += 1

            object.append(count)
            objects.append(object)
        f.close()

    tokens = dict() #ключ - id, значение - структура d
    for path_i in paths[1]:
        #файл tok----------------------------------------------------------
        f = open(path_i,'r')
        d = dict()
        for line in f:

            arr = line.split(" ")
            if arr[0] != '\n':
                if arr[3][-1] == "\n":
                    arr[3] = arr[3][0:-1]
                d['txt'] = arr[3]
                d['class'] = 0
                d['obj_id'] = 0
                d['tokens_num'] = 0
                tokens[int(arr[0])] = d.copy()
        f.close()

    max_object_size = 0

    result = []
    for obj in objects:
        num = int(obj[-1])
        if num > max_object_size:
            max_object_size = num
        tokensList = []
        d = dict()
        d['class'] = obj[1]
        for spanId in range(2, num + 2, 1):
            #d = dict()
            #d['class'] = obj[1]
            #tokensList = []
            for tokenId in (spans[str(obj[spanId])]['tokens']):

                if tokens[tokenId]['class'] < 2 or tokens[tokenId]['tokens_num'] < num :
                    tokensList.append(tokens[tokenId])
                    tokens[tokenId]['class'] = obj[1]
                    tokens[tokenId]['tokens_num'] = num
                    tokens[tokenId]['obj_id'] = obj[0]
        d['tokens'] = tokensList
        if(d['class'] >=2 and len(tokensList)!=0):
            result.append(d.copy())

    # NoNERnum = math.floor(len(result) / 3)
    # #print("NoNERnum", NoNERnum)
    # i = 0
    # NoNER = []
    # #while (i < NoNERnum):
    # for j in tokens.keys():
    #         x = []
    #         if tokens[j]["class"] == 0:
    #             count1 = 0
    #             x.append(tokens[j]['txt'])
    #             tokens[j]['class'] = 5
    #             k = j+1
    #             i += 1
    #             NoNERRandNum = random.randint(1,2)
    #             while (count1 < NoNERRandNum) and (tokens.get(k) is not None) and (tokens[k]['class'] == 0):
    #                 x.append(tokens[k]['txt'])
    #                 tokens[k]['class'] = 5
    #                 k += 1
    #                 count1 += 1
    #             NoNER.append(x)
    #             d = dict()
    #             d['tokens'] = x
    #             d['class'] = 5
    #             result.append(d.copy())
    #         if i > NoNERnum: break
    f = open('check_objects', 'w')
    for t in result:
        print(t,file=f)
    return result

def sentenseToCRFParams(sentenses):
    #print(sentenses[0])
    CRF_x = []
    CRF_y = []
    for sentense in sentenses:
        tokens = [token['txt'] for token in sentense]
        CRF_x.append(tokens)

        labels = []
        # Labels codes
        # 0 - OUT
        # 10 - Unary
        # 13 - Begin
        # 14 - Inner
        # 15 - Last



        state = 0
        for token in sentense:
            if(state == 0):
                if(token['class'] < 2):
                    labels.append(OUT)
                else:
                    if(token['tokens_num'] == 1):
                        labels.append(UNARY)
                    else:
                        state = 1
                        labels.append(BEGIN)
            else:
                if(token['class'] < 2):
                    labels[-1] = LAST
                    labels.append(OUT)
                    state = 0
                else:
                    labels.append(INNER)


        CRF_y.append(labels)
    res = dict()
    res['CRF_X'] = CRF_x
    res['CRF_Y'] = CRF_y
    return res

def getObjIds(sentenses):
    result = []
    for s in sentenses:
        for token in s:
            id = token['obj_id']
            if not id in result:
                result.append(id)
    return result
def splitTokens(tokens):
    s = ""
    for t in tokens:
        s += t+' '
    return s
def checkResult(checkList, resultList, sentenses):
    total = 0
    match = 0
    founded = 0
    index = 0
    ORG = 2
    PER = 3
    LOC = 4
    total_person = 0
    match_person = 0
    total_location = 0
    match_location = 0
    total_org = 0
    match_org = 0
    matchedEntities = dict()
    matchedEntities[ORG] = 0
    matchedEntities[PER] = 0
    matchedEntities[LOC] = 0
    foundedEntities = dict()
    foundedEntities[ORG] = 0
    foundedEntities[PER] = 0
    foundedEntities[LOC] = 0
    f = open("dif_res.txt","w")
    if len(checkList) != len(result_list):
        print('fail compared sets')
        return -1
    for sentense in checkList:
        #print(sentense)
        #print(index)
        total += len(sentense)
        founded += len(result_list[index])

        f.write('sentense number '+str(index)+' CHECKED length = '+str(len(sentense))+
            ' FOUNDED length = '+str(len(result_list[index]))+'\n')
        f.write(splitTokens(sentenses[index])+'\n')
        f.write('not founded from CHECKED:\n')

        checkedSpans = []
        s1 = ''
        tmp_foundedSpans = resultList[index].copy()
        for checkEnt in sentense:
            checkStr = splitTokens([t['txt'] for t in checkEnt['tokens']])
            if checkEnt['class'] == ORG:
                total_org += 1
            elif checkEnt['class'] == PER:
                total_person += 1
            elif checkEnt['class'] == LOC:
                total_location += 1

            checkedSpans.append(checkStr)
            isNotFound = True
            for iSpans in range(0, len(tmp_foundedSpans)):
                if checkStr == tmp_foundedSpans[iSpans]['txt']: #and checkEnt['class'] == res['class']:
                    foundedEntities[tmp_foundedSpans[iSpans]['class']]+=1
                    match += 1
                    if checkEnt['class'] == tmp_foundedSpans[iSpans]['class']:
                        matchedEntities[tmp_foundedSpans[iSpans]['class']] += 1
                    isNotFound = False
                    tmp_foundedSpans.pop(iSpans)
                    break
            if isNotFound:
                f.write(checkStr+'  |')
            else:
                s1 += checkStr+'  |'
        f.write('\n')
        f.write('MATCHED:\n')
        f.write(s1+'\n')
        f.write('founded MORE:\n')
        foundedSpans = [res['txt'] for res in resultList[index]]
        for span in foundedSpans:
            isNotFound = True
            for s in checkedSpans:
                if span == s:
                    isNotFound = False
            if isNotFound:
                f.write(span+'  |')

        f.write('\n')
        f.write('\n')
        index += 1
    d = dict()
    d['total'] = total
    d['match'] = match
    d['founded'] = founded
    d['total_person'] = total_person
    d['founded_person'] = foundedEntities[PER]
    d['match_person'] = matchedEntities[PER]
    d['total_org'] = total_org
    d['founded_org'] = foundedEntities[ORG]
    d['match_org'] = matchedEntities[ORG]
    d['total_loc'] = total_location
    d['founded_loc'] = foundedEntities[LOC]
    d['match_loc'] = matchedEntities[LOC]
    f.close()
    return d
def processTokens(Listtokens):
    result = []
    for tokens in Listtokens:
        l = [t['txt'] for t in tokens ]
        result.append(l)
    #print(result)
    return result
def getEntities(sentenses):
    ListEntities = []
    for sentense in sentenses:
        entities = []
        state = 0
        d = dict()
        for token in sentense:
            if(state == 0):
                if(token['class'] >= 2):
                    if(token['tokens_num'] == 1):
                        d.clear()
                        d['tokens'] = []
                        d['tokens'].append(token)
                        d['class'] = token['class']
                        entities.append(d.copy())
                    else:
                        state = 1
                        d.clear()
                        d['tokens'] = []
                        d['tokens'].append(token)
                        d['class'] = token['class']
            else:
                if(token['class'] < 2):
                    entities.append(d.copy())
                    state = 0
                else:
                    d['tokens'].append(token)
        ListEntities.append(entities)
    return ListEntities

if __name__ == '__main__':
    # path_tok = '/home/anastasiya/Dialog/factRuEval-2016-master/devset/book_93.tokens'
    # path_obj = '/home/anastasiya/Dialog/factRuEval-2016-master/devset/book_93.objects'
    # path_span = '/home/anastasiya/Dialog/factRuEval-2016-master/devset/book_93.spans'
    path = "/home/anastasiya/Dialog/devset"

    entities = retriveNamedEntity(path)
    #for e in entities:
         #print(e['tokens'], e['class'])
    markedSentense = getTokens(path)

    #self.classifierSVM.fit(x_train, y_train)

    #for sentense in markedSentense[:2]:
    #    for tokenId in sentense.keys():
    #        print(sentense[tokenId])

    # shapes = []
    # for sentense in markedSentense:
    #     shapes += ([ getShape(token['txt']) for token in sentense])
    # leShape.fit(shapes)
    # print(leShape.classes_)
    #

    bound = int(len(markedSentense)*0.7)
    tmp_train = markedSentense[:bound]
    tmp_test = markedSentense[bound:]
    tmp_test_Entities = getEntities(tmp_test)

    # print('================================test entities begin===============================')
    # for sentense in tmp_test_Entities:
    #     for span in sentense:
    #         print([t['txt'] for t in span['tokens']])
    # print('================================test entities END=================================')
    test_sent = sentenseToCRFParams(tmp_test)

    entities = retriveNamedEntity(path)

    listIds = getObjIds(tmp_test)

    train_entities = []
    test_entities = []
    for e in entities:
        #print(e['tokens'][0])
        if e['tokens'][0]['obj_id'] in listIds:
            train_entities.append(e)
        else:
            test_entities.append(e)

    detector = NamedEntityRecognition()
    print('======================================= train SVM================================')

    x_train_entity = [e['tokens'] for e in train_entities]
    y_train_entity = [e['class'] for e in train_entities]

    x_test_entity = [e['tokens'] for e in test_entities]
    y_test_entity = [e['class'] for e in test_entities]

    y_train_entity = [y-2 for y in y_train_entity]
    #print (y_train)
    #from sklearn import preprocessing
    le = LabelEncoder()
    le.fit(y_train_entity)
    print(le.classes_)
    # print(le.transform(-1))
    # print(le.transform(0))
    # print(le.transform(1))
    # print(le.transform(2))

    #select type of classifier: classifierMNB for Naive Bayes, classifierSVM for SVM; classifierMaxEnt_LogReg for MaxEnt
    Classifier = detector.classifierMNB
    Classifier.fit(processTokens(x_train_entity), y_train_entity)

    print('========================================end train svm============================')

    #print(tmp_train[0])
    tmp_train = sentenseToCRFParams(tmp_train)
    tmp_test1 = sentenseToCRFParams(tmp_test)
    x_train_sent = tmp_train['CRF_X']
    y_train_sent = tmp_train['CRF_Y']

    x_test_sent = tmp_test1['CRF_X']
    y_test_sent = tmp_test1['CRF_Y']

    print('======================================== train CRF============================')

    tmp = ExtractFeaturesToArray()
    print(detector.classifierCRF)
    print('кол-во признаков ',len(tmp.info(['','','','',''])))

    print(len(x_train_sent))
    print(len(y_test_sent))

    detector.classifierCRF.fit(x_train_sent, detector.convertYtoCRF(y_train_sent))

    print('========================================end train CRF============================')

    print('=====================================START of segmentation============================')
    CRF_result = detector.classifierCRF.predict(x_test_sent)
    print('=====================================END of segmentation  ============================')

    # for i in range(0, 20):
    #     print('======================')
    #     print(np.array(y_test_sent[i]))
    #     print(CRF_result[i])


    x_spansList = getSpans(x_test_sent, CRF_result)

    print('=====================================Clasify test data============================')
    result_list = []
    f = open('before_classify','w')
    for spans in x_spansList:
        sentense = []
        for span in spans:
            d = dict()
            s = []
            s.append(span)
            tmp = Classifier.predict(s)[0]
            if(tmp < 0):
                print('ALLERT!!!!!!!!!!!!!!!!!')
            d['class'] = tmp+2
            d['txt'] = splitTokens(span)
            f.write(d['txt']+' '+str(d['class'])+'\n')
            sentense.append(d.copy())
        result_list.append(sentense)
    #   Classifier_result.append(detector.classifierSVM.predict(spans))

    #Classifier_result = y_test_entity
    print('=================================END Clasify test data============================')

    labels = ['N', 'N', 'Organization', 'Person', 'Location', 'Other']

    # result_list = []
    # for spans in x_spansList:
    #     for index in range(0, len(spans)):
    #         d = dict()
    #         d['txt'] = splitTokens(x_spans[index])
    #         d['class'] = Classifier_result[index]+2
    #         result_list.append(d.copy())
    res = checkResult(tmp_test_Entities, result_list, x_test_sent)
    print('total: ',res['total'],'match: ',res['match'],'founded: ',res['founded'])
    print('total_person: ',res['total_person'],'match_person: ',res['match_person'],'founded_person: ',res['founded_person'])
    print('total_loc: ',res['total_loc'],'match_loc: ',res['match_loc'],'founded_loc: ',res['founded_loc'])
    print('total_org: ',res['total_org'],'match_org: ',res['match_org'],'founded_org: ',res['founded_org'])
    #detector.train_N0CRF(result_x, result_y)

    print("end!")