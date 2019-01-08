import csv
import random as rnd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


def readCSV(pathToFile):
    file = open(pathToFile, mode = 'r')
    reader = csv.DictReader(file)
    points = []
    for line in reader: 
        points.append((float(line['log']), 
        float(line['lat']), 
        float(line['request_ts']), 
        float(line['trans_ts']), 
        line['label'], 0))
    return points


def reduction(points):
    log = []
    lat = []
    request_ts = []
    trans_ts = []
    for point in points:
        log.append(point[0])
        lat.append(point[1])
        request_ts.append(point[2])
        trans_ts.append(point[3])
    min_log = min(log)
    min_lat = min(lat)
    min_ts = min(min(request_ts), min(trans_ts))
    newPoints = []
    for point in points:
        newPoints.append((point[0] - min_log,
        point[1] - min_lat,
        point[2] - min_ts,
        point[3] - min_ts,
        point[4],
        point[5]))
    return newPoints


def definePointRegion(points):
    definedPoints = []
    for point in points:
        if point[0] <= 0.07738:
            region = 0
        elif 0.17933 < point[0]:
            region = 1
        elif 0.07738 < point[0] <= 0.101 and 0.01996 < point[1]:
            region = 2
        elif 0.07738 < point[0] <= 0.101 and point[1] <= 0.01415:
            region = 3
        elif 0.07738 < point[0] <= 0.09316 and 0.01415 < point[1] <= 0.01996:
            region = 4
        elif 0.09316 < point[0] <= 0.101 and 0.01415 < point[1] <= 0.01996:
            region = 5
        else:
            region = 6
        definedPoints.append((point[0],
        point[1],
        point[2],
        point[3],
        point[4],
        region))
    return definedPoints


def getUnknownPoints(points):
    unknownPoints = []
    for point in points:
        if point[4] == '?':
            unknownPoints.append(point)
    return unknownPoints


def defineRoute(unknownPoints):
    answers = []
    for point in unknownPoints:
        answers.append(definePointRoute(point))
    return answers


def getTrainData(points, numRegion):
    trueTrain = []
    trueAnswers = []
    for point in points:
        if (point[4] == '0' or point[4] == '1' or point[4] == '2') and point[5] == numRegion:
            trueTrain.append((point[0], point[1], point[2], point[3]))
            trueAnswers.append(point[4])
    return trueTrain, trueAnswers


points = readCSV("transport_data.csv")
newPoints = reduction(points)
definedPoints = definePointRegion(newPoints)

unknownPoints = getUnknownPoints(definedPoints)

train5, answers5 = getTrainData(definedPoints,5)
train6, answers6 = getTrainData(definedPoints,6)
# x_train5, x_test5, y_train5, y_test5 = train_test_split(train5, answers5, test_size=0.1, random_state=42)
# x_train6, x_test6, y_train6, y_test6 = train_test_split(train6, answers6, test_size=0.1, random_state=42)
# model5 = ExtraTreesClassifier(n_estimators = 10000, random_state=90)
model5 = RFE(RandomForestClassifier(n_estimators = 10000, random_state=90),n_features_to_select=15)
# model6 = ExtraTreesClassifier(n_estimators = 10000, random_state=90)
model6 = RFE(RandomForestClassifier(n_estimators = 10000, random_state=90),n_features_to_select=15)
model5.fit(train5,answers5)
model6.fit(train6,answers6)
# model5.fit(x_train5,y_train5)
# model6.fit(x_train6,y_train6)

# print('region 5: ' + str(model5.score(x_test5, y_test5)))
# print('dimention 5: ' + str(len(x_test5)))
# print('region 6: ' + str(model6.score(x_test6, y_test6)))
# print('dimention 6: ' + str(len(x_test6)))

def definePointRoute(point):
    if point[5] == 0:
        return 1
    if point[5] == 1:
        return 2
    if point[5] == 2:
        return 2
    if point[5] == 3:
        return 0
    if point[5] == 4:
        return 1
    if point[5] == 5:
        return model5.predict([(point[0],point[1],point[2],point[3])])[0]
    if point[5] == 6:
        return model6.predict([(point[0],point[1],point[2],point[3])])[0]


predicts = defineRoute(unknownPoints)


file = open("answers.txt", mode='w')
for item in predicts:
    file.write(str(item) + "\n")
file.close()