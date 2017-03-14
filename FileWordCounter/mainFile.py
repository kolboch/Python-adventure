#simple file handling and counting word occurences

import sys
import operator

#gets file from given filename
def getFile(filename):
    f = open(filename, 'rU')
    return f

#closes given file
def closeFile(file):
    file.close()

#return dictionary with key value pairs like word - counter
def countWords(file):
    dictionary = {}
    for wordLine in file:
        word_list = wordLine.split()
        for word in word_list:
            if word in dictionary:
                dictionary[word] = dictionary[word] + 1
            else:
                dictionary[word] = 1
    return dictionary

#method for printing dictionary in format key: x value: y /n - for each pair
def printDictionary(dictionary):
    for k in dictionary.keys():
        print('key: {} value: {}'.format(k, dictionary[k]))

#prints list, each record new line
def printList(list):
    for elem in list:
        print(elem)

#prints dictionary like key value pairs in sorted by values order
def printDictSortedByValues(dictionary):
    by_values_tuples = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    for k, v in by_values_tuples:
        print('{}:{} {}:{}'.format('key', k, 'value', v))

#main
def main():
    file = getFile('text.txt')
    dictionary = countWords(file)
    printDictSortedByValues(dictionary)
    closeFile(file)

if __name__ == '__main__':
    main()