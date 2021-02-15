#!/usr/bin/env python3

'''
Baseline: Read in training data
          use unigram.Unigram to train model
Author: Jacob Huber
'''
import unigram
import five_gram
import sys

filename_train = "data/train"
filename_dev = "data/dev"
filename_test = "data/test"

def read_data():
    ''' Read in data from data/train and strip newlines '''
    with open(filename_train) as t:
        content = t.read().splitlines()
    
    return content

def predict(model, data):
    '''Read in data and test the accuracy of the model'''
    cur_state = model.start()
    total = 0
    correct = 0

    with open(data) as d:
        lines = d.read().splitlines()
    
    timer = 20
    for line in lines:
        cur_state = model.start()
        for c in list(line) + ['<EOS>']:
            prediction = model.best(cur_state)
            total += 1
            if prediction == c:
                correct += 1
            if timer > 0:    
                #print(f'c = {c}')
                #print(f'p = {prediction}') 
                #print(f'correct = {correct}') 
                #print(f'total = {total}') 
                timer = timer - 1
            cur_state = model.read(cur_state, c)

    return correct/total

def main(option):
    if option == 1 or option == 3:
        model = unigram.Unigram(read_data())
        print('-------------- Baseline -----------------')
        print(f'Accuracy of Unigram on data/dev = {predict(model, filename_dev)*100:.2f} %\n')
    if option == 2 or option == 3:
        five_gram_model = five_gram.Five_gram(read_data())
        print('-------- n-gram Language Model ----------')
        print(f'Accuracy of 5-Gram on data/dev = {predict(five_gram_model, filename_dev)*100:.2f} %')
        print(f'Accuracy of 5-Gram on data/test = {predict(five_gram_model, filename_test)*100:.2f} %')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./work.py [OPTION]\n  -unigram\n  -fivegram\n  -both")
    else:
        model = 0
        if sys.argv[1][1:] == "unigram":
            model = 1
        elif sys.argv[1][1:] == "fivegram":
            model = 2
        elif sys.argv[1][1:] == "both":
            model = 3
        
        if model > 0:
            main(model)
        else:
            print("invalid option")

