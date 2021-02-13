#!/usr/bin/env python3

'''
Baseline: Read in training data
          use unigram.Unigram to train model
Author: Jacob Huber
'''
import unigram

filename_train = "data/train"
filename_dev = "data/dev"

def read_data():
    ''' Read in data from data/train and strip newlines '''
    with open(filename_train) as t:
        content = t.read().splitlines()
    
    return content

def predict(model):
    '''Read in data/dev and test the accuracy of the unigram model'''
    cur_state = model.start()
    total = 0
    correct = 0

    with open(filename_dev) as d:
        lines = d.read().splitlines()
        
    for line in lines:
        for c in list(line) + ['<EOS>']:
            prediction = model.best(cur_state)
            total += 1
            if prediction == c:
                correct += 1
            #print(f'c = {c}')
            #print(f'p = {prediction}') 
            #print(f'correct = {correct}') 
            #print(f'total = {total}') 
            
            cur_state = model.read(cur_state, c)

    return correct/total

def main():
    model = unigram.Unigram(read_data())
    print('-------------- Baseline -----------------')
    print(f'Accuracy of Unigram on data/dev = {predict(model)*100:.2f} %')
    
    


if __name__ == '__main__':
    main()

