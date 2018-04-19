import sys
import math

# get entropy value
def entropy(attributes, data, target):
    index = attributes.index(target)
    freqDict = dict()
    dataEntropy = 0.0

    # count target values
    for tuple in data:
        if tuple[index] in freqDict:
            freqDict[tuple[index]] += 1
        else:
            freqDict[tuple[index]] = 1

    # calculate total entropy
    for freq in freqDict.values():
        dataEntropy += (-freq / len(data)) * math.log2(freq / len(data))

    return dataEntropy


# get informationGain
def informationGain(attributes, data, attr, target):
    freqDict = dict()
    subEntropy = 0.0
    infoGain = 0.0
    index = attributes.index(attr)

    # count 'attr' values
    for tuple in data:
        if tuple[index] in freqDict:
            freqDict[tuple[index]] += 1
        else:
            freqDict[tuple[index]] = 1

    # for each value
    for value in freqDict.keys():
        prob = freqDict[value] / len(data)  # probability that 'value' appears
        dataSubset = [entry for entry in data if entry[index] == value]  # a subset of data that has 'value'
        subEntropy += prob * entropy(attributes, dataSubset, target) # calculate subentropy

    infoGain = (entropy(attributes, data, target) - subEntropy) # calculate information gain

    return infoGain


# choose the best attribute when dataset, attributeset and target attribute was given
def chooseBestAttr(data, attributes, target):
    best = attributes[0]
    maxGain = 0;

    for attr in attributes[:-1]: # [:-1] to except label attribute
        tempGain = informationGain(attributes, data, attr, target)
        if tempGain > maxGain: # iterate to find best attribute which has maximum information gain
            maxGain = tempGain
            best = attr

    return best


# when dataset and attributeset was given, get whole values which 'attr' has
def getValues(data, attributes, attr):
    index = attributes.index(attr)
    values = list()

    for tuple in data:
        if tuple[index] not in values:
            values.append(tuple[index])

    return values


# get new dataset
def getNewData(data, attributes, best, value):
    newDatas = list()
    index = attributes.index(best)

    for tuple in data:
        if (tuple[index] == value):
            newTuple = list()
            for i in range(len(tuple)):
                if (i != index):
                    newTuple.append(tuple[i])
            newDatas.append(newTuple)

    return newDatas


# make a Decision Tree
def makeDecisionTree(data, attributes, target, default):
    freqDict = dict()
    index = attributes.index(target
                             )
    for tuple in data:
        if tuple[index] in freqDict:
            freqDict[tuple[index]] += 1
        else:
            freqDict[tuple[index]] = 1

    if not data or len(attributes) <= 1: # if there is no more training data or no more attribute stop
        return default
    elif len(freqDict.keys()) == 1: # if all label are same, return that label
        return list(freqDict.keys())[0]
    else:
        best = chooseBestAttr(data, attributes, target) # find the best attribute to split
        tree = {best: {}}

        # given best attribute, make tree recursively with new dataset and new attributeset
        for value in getValues(data, attributes, best):
            newData = getNewData(data, attributes, best, value)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeDecisionTree(newData, newAttr, target, default)
            tree[best][value] = subtree

    return tree


# overall process to make a Decision Tree model
def makeDTModel():
    data = []
    with open(sys.argv[1], 'r') as f:
        input = f.readlines()

    for line in input:
        line = line.strip("\r\n")
        data.append(line.split('\t'))

    attributes = data[0]
    data.remove(attributes)
    target = attributes[-1]
    labels = {}
    index = attributes.index(target)

    # make a frequent list of label
    for tuple in data:
        if (tuple[index] in labels):
            labels[tuple[index]] += 1
        else:
            labels[tuple[index]] = 1
    max = 0
    default = str()
    # find default value (the most frequent label value)
    for key in labels.keys():
        if labels[key] > max:
            max = labels[key]
            default = key

    tree = makeDecisionTree(data, attributes, target, default)

    return attributes, tree, default

# classifiy the test set with decision tree
def classify(attributes, tree, default):
    with open(sys.argv[2], 'r') as f:
        test = f.readlines()
    testSet = list()

    for line in test:
        testSet.append(line.strip('\n'))
    testSet.pop(0)
    resultSet = []

    for testTuple in testSet:
        test = testTuple.split()
        next = tree

        while True:
            # if there is no more child node stop
            if not isinstance(next, dict):
                break
            attr = list(next.keys())
            attr = attr[0]
            index = attributes.index(attr)
            next = next[attr]
            if test[index] in next:
                next = next[test[index]]
            else:
                next = default
                break
        resultSet.append(testTuple+'\t'+next)

    # make an output string
    output = str()
    for i in range(len(attributes)-1):
        output += attributes[i] + '\t'
    output += attributes[len(attributes)-1] + '\n'
    for result in resultSet:
        output += result + '\n'

    with open(sys.argv[3], 'w') as f:
        f.write(output)


def main():
    attributes, tree, default = makeDTModel()
    classify(attributes, tree, default)


if __name__ == '__main__':
    main()
