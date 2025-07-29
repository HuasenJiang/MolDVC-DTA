from dataproc import *

import os
from init_params import argparser, logging
from copy import deepcopy
from model import CSatDTAmodel

figdir = "figures/"


def nfold_1_2_3_setting_sample(XD, XT,  Y, label_row_inds, label_col_inds, measure, FLAGS, dataset):

    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    
    foldinds = len(outer_train_sets) # this gives number of train sets. so no of k fold = len(outer_train_set) + 1 (test set)
    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []
    if(FLAGS.problem_type == 1): #if data is not folded already
        x = outer_train_sets
        r1 = int(0.8 * len(x))
        r2 = int(0.1 * len(x))
        x1 = x[0:r1]
        x2 = x[r1:r1 + r2]
        x3 = x[r1 + r2:]

        train_sets.append(x1)
        val_sets.append(x2)
        test_sets.append(x3)
 
    else:
        #if folds are already maintained in data
        for val_foldind in range(foldinds): 
            val_fold = outer_train_sets[val_foldind]
            val_sets.append(val_fold)
            otherfolds = deepcopy(outer_train_sets)
            otherfolds.pop(val_foldind)
            otherfoldsinds = [item for sublist in otherfolds for item in sublist]
            train_sets.append(otherfoldsinds)
            test_sets.append(test_set)
            print("val set", str(len(val_fold)))
            print("train set", str(len(otherfoldsinds)))



    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = CSatDTAmodel(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, FLAGS, train_sets, val_sets)
   
    bestparam, best_param_list, bestperf, all_predictions, all_losses = CSatDTAmodel(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, FLAGS, train_sets, test_sets)

    
    testperf = all_predictions[bestparamind]##pointer pos 

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" % 
            (bestparamind, bestparam), FLAGS)


    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd


def experiment(FLAGS, perfmeasure, foldcount=5): #5-fold cross validation + test

    dataset = DataSet( fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                      setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE: data can brokendown in different ways in fold. Its just the type number
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)  #basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                     perfmeasure, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    print("S1_avgperf", S1_avgperf, "S1_avgloss", S1_avgloss, "S1_teststd", S1_teststd)