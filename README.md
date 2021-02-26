## CSE 40657/60657
# Homework 1
# Jake Huber (jhuber4)

URL: https://github.com/huber02j/nlp-hw1

How to build/run for first 2 parts:

    work.py has the code to test the ngram and unigram models 
    Usage: ./work.py [OPTION]
      -unigram      trains unigram model and tests accuracy on data/dev
      -fivegram     trains 5-gram model and tests data/dev and data/test
      -both         trains and tests both models

    Part 1:
        run the program by typing ./work.py -unigram
        This will run the program to test the unigram model and output the accuracy when tested on data/dev
        The accuracy when run equals 16.48%

    Part 2:
        run the program by typing ./work.py -fivegram

        The accuracy on data/dev  = 49.49%
        The accuracy on data/test = 49.75%

    Part 3:
How to build/run for part 3:
        run the program by typing: python3 rnn.py [OPTION]
          -mini     trains on data/minitrain and tests with data/dev
          -train    trains on data/train and tests with data/dev
          -test     test with data/test on saved model
        
        Train: data/minitrain; Test: data/dev;
            time=86.32948660850525 train_ppl=20.244939610330746 dev_acc=0.23343373493975902
            time=83.58734726905823 train_ppl=13.602038292603215 dev_acc=0.2499234225035736
            time=87.24636125564575 train_ppl=11.705748801437947 dev_acc=0.2568919746783745
            time=84.29236078262329 train_ppl=10.182263276396135 dev_acc=0.2608229528282622
            time=90.42376446723938 train_ppl=8.875126662471116 dev_acc=0.2673320400245048
            time=93.44291090965271 train_ppl=7.948933083049995 dev_acc=0.28234122932407596
            time=88.48179578781128 train_ppl=7.2470987399021585 dev_acc=0.31460588115172555
            time=85.77874898910522 train_ppl=6.720204664033403 dev_acc=0.314503777823157
            lr=0.05
            time=93.3861951828003 train_ppl=6.220315122230227 dev_acc=0.3171074127016541
            time=87.81322526931763 train_ppl=6.010960039034366 dev_acc=0.32313150908719623
            time=94.39808893203735 train_ppl=5.83355665309496 dev_acc=0.32211047580151114
            lr=0.025
            time=93.07654356956482 train_ppl=5.616404059888026 dev_acc=0.3287727179906065
            time=87.79611730575562 train_ppl=5.535888691976268 dev_acc=0.3280579946906269
            lr=0.0125
            time=86.25437521934509 train_ppl=5.42740904375427 dev_acc=0.3303553195834184
            time=87.41259264945984 train_ppl=5.38662323845803 dev_acc=0.3315039820298142
            time=87.88374900817871 train_ppl=5.348417831403773 dev_acc=0.33193792117623033
            time=88.68090724945068 train_ppl=5.310827857814372 dev_acc=0.33173371451909334
            lr=0.00625

            Final train_ppl = 5.31
            Final dev_acc = 0.33

        Train: data/train; Test: data/dev;
           ----- Training with train, Testing with devdata -----
            time=4024.4254343509674 train_ppl=8.497487311086585 dev_acc=0.48187665917908923
            time=4309.674641370773 train_ppl=6.467769501404308 dev_acc=0.4998213191750051
            time=4303.628115653992 train_ppl=6.155121746948871 dev_acc=0.5002807841535634
            time=4247.474761247635 train_ppl=5.9951735095995495 dev_acc=0.5030120481927711
            time=4334.59267783165 train_ppl=5.8988465606144835 dev_acc=0.5131458035531958
            time=4349.490671634674 train_ppl=5.828443345348578 dev_acc=0.5086787829283235
            lr=0.05
            time=4274.227051734924 train_ppl=5.614181577598505 dev_acc=0.5125331835817848
            time=4324.529010772705 train_ppl=5.565524545650352 dev_acc=0.5140902593424546
            time=4363.456878423691 train_ppl=5.536793497525323 dev_acc=0.5156728609352665
            time=4226.080573320389 train_ppl=5.512364500174756 dev_acc=0.5132479068817644
            lr=0.025
            time=4332.84299826622 train_ppl=5.402206173724791 dev_acc=0.5196038390851542 
            
            Final dev_acc = 0.5196 

        Model run on data/test. Accuracy = ;
            ----- Testing with testdata -----
            dev_acc=0.5360408652330201

