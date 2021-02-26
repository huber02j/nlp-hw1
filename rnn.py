#!/usr/bin/env python3

'''
Part 3: RNN language model
Author: Jake Huber (jhuber4)
'''

import collections, time, math, random
import torch
import copy
import sys

if torch.cuda.device_count() > 0:
    print(f'Using GPU ({torch.cuda.get_device_name(0)})')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'

def read_data(filename):
    return [list(line.rstrip('\n')) + ['<EOS>'] for line in open(filename)]

minidata = read_data('data/minitrain')
traindata = read_data('data/train')
devdata = read_data('data/dev')
testdata = read_data('data/test')

class Vocab:
    def __init__(self, counts, size):
        if not isinstance(counts, collections.Counter):
            raise TypeError('counts must be a collections.Counter')
        words = {'<EOS>', '<UNK>'}
        for word, _ in counts.most_common():
            words.add(word)
            if len(words) == size:
                break
        self.num_to_word = list(words)    
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}

    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        return self.num_to_word[num]


def logprobs(D, h, e):
    #computing y
    return torch.log_softmax(torch.add(D @ h, e), dim=0)

def main(train, dev):
    vocab_size = 100
    d = 64
    
    chars = collections.Counter()
    for line in train:
        chars.update(line)
    vocab = Vocab(chars, vocab_size) # For our data, 100 is a good size.
    #print(vocab.word_to_num)
    
    # Parameters
    h_0 = torch.normal(mean=0, std=0.01, size = (d,), requires_grad=True, device=device)
    W = torch.normal(mean=0, std=0.01, size = (d, d, d), requires_grad=True, device=device)
    B = torch.normal(mean=0, std=0.01, size = (vocab_size, d), requires_grad=True, device=device)
    c = torch.normal(mean=0, std=0.01, size = (d,), requires_grad=True, device=device)
    D = torch.normal(mean=0, std=0.01, size = (vocab_size, d), requires_grad=True, device=device)
    e = torch.normal(mean=0, std=0.01, size = (vocab_size,), requires_grad=True, device=device)

    params = [h_0, W, B, c, D, e]

    optim = torch.optim.SGD(params, lr=0.1)

    prev_dev_acc = None #previous dev accuracy

    for epoch in range(20): 
        epoch_start = time.time()

        # Run through the training data

        random.shuffle(train) # Important

        train_loss = 0  # Total negative log-probability
        train_chars = 0 # Total number of characters
        for chars in train:
            # Compute the negative log-likelihood of this line,
            # which is the thing we want to minimize.
            
            # pass in D, h, and e
            loss = -logprobs(params[4], params[0], params[5])[vocab.numberize(chars[0])]
            h = params[0]
            for index, c in enumerate(chars[:-1]):
                target = chars[index+1]

                # compute h vector
                v = params[2][vocab.numberize(c)]
                z = torch.einsum('jkl,k,l->j', params[1], h, v)
                h = torch.tanh(torch.add(z, params[3]))

                train_chars += 1
                loss -= logprobs(params[4], h, params[5])[vocab.numberize(target)]
                
            # Keep a running total of negative log-likelihood.
            # The .item() turns a one-element tensor into an ordinary float,
            # including detaching the history of how it was computed,
            # so we don't save the history across sentences.
            train_loss += loss.item()

            # Compute gradient of loss with respect to parameters.
            optim.zero_grad()   # Reset gradients to zero
            loss.backward() # Add in the gradient of loss

            # Clip gradients (not needed here, but helpful for RNNs)
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            # Do one step of gradient descent.
            optim.step()
            
        # Run through the development data
        
        dev_chars = 0   # Total number of characters
        dev_correct = 0 # Total number of characters guessed correctly
        for chars in dev:
            h = params[0]
            for index, c in enumerate(chars[:-1]):
                dev_chars += 1
                target = chars[index+1]
                # Compute tensors
                v = params[2][vocab.numberize(c)] 
                z = torch.einsum('jkl,k,l->j', params[1], h, v)
                h = torch.tanh(torch.add(z, params[3]))

                # Find the character with highest predicted probability.
                # The .item() is needed to change a one-element tensor to
                # an ordinary int.
                best = vocab.denumberize(logprobs(params[4], h, params[5]).argmax().item())
                #print(f'{c} : guess = {best}')
                if best == target:
                    dev_correct += 1


        dev_acc = dev_correct/dev_chars
        print(f'time={time.time()-epoch_start} train_ppl={math.exp(train_loss/train_chars)} dev_acc={dev_acc}')

        # Important: If dev accuracy didn't improve, halve the learning rate
        if prev_dev_acc is not None and dev_acc <= prev_dev_acc:
                optim.param_groups[0]['lr'] *= 0.5
                print(f"lr={optim.param_groups[0]['lr']}")

        # When the learning rate gets too low, stop training
        if optim.param_groups[0]['lr'] < 0.01:
            break

        prev_dev_acc = dev_acc
        
        par_voc = [params, vocab]
        torch.save(par_voc, 'model')
        
def test():
    
    prev_dev_acc = None #previous dev accuracy

    par_voc = torch.load('model')
    params = par_voc[0]
    vocab = par_voc[1]

    dev_chars = 0   # Total number of characters
    dev_correct = 0 # Total number of characters guessed correctly
    for chars in testdata:
        h = params[0]
        for index, c in enumerate(chars[:-1]):
            dev_chars += 1
            target = chars[index+1]
            # Compute tensors
            v = params[2][vocab.numberize(c)] 
            z = torch.einsum('jkl,k,l->j', params[1], h, v)
            h = torch.tanh(torch.add(z, params[3]))

            # Find the character with highest predicted probability.
            # The .item() is needed to change a one-element tensor to
            # an ordinary int.
            best = vocab.denumberize(logprobs(params[4], h, params[5]).argmax().item())
            #print(f'{c} : guess = {best}')
            if best == target:
                dev_correct += 1


    dev_acc = dev_correct/dev_chars
    print(f'dev_acc={dev_acc}')




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 rnn.py [OPTION]\n  -mini\n  -train\n  -test")
    else:
        if sys.argv[1][1:] == "mini":
            print('----- Training with minitrain, Testing with devdata -----')
            main(minidata, devdata)
        elif sys.argv[1][1:] == "train":
            print('----- Training with train, Testing with devdata -----')
            main(traindata, devdata)
        elif sys.argv[1][1:] == "test":
            print('----- Testing with testdata -----')
            test()
        else:
            print("invalid option")

