from init_params import argparser, logging
import numpy as np
from interaction import *
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from atten import *
from figplot import *
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys

def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select

def CSatDTAmodel(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi/filter size
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384] - filter size for proteins/target
    epoch = FLAGS.num_epoch                                 #100
    batchsz = FLAGS.batch_size                             #256

    logging("---Parameter Search-----", FLAGS)


    w = len(val_sets)
    
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)] 


    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]


        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)



        pointer = 0
        
        for param1ind in range(len(paramset1)): #hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)): #learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                        param3value = paramset3[param3ind]
        
                        XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
                        XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
                        
                        NUM_FILTERS = param1value #64 #96
                        FILTER_LENGTH1 = param2value #3 #8
                        FILTER_LENGTH2 = param3value #3 #12


                        encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
                        
                        encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
                        encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)

                        x1 = (encode_smiles.shape[1])
                        y1 = (encode_smiles.shape[2])

                        x_smiles = augmented_conv1d(encode_smiles, shape = (x1, y1), filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,
                                        strides = 1,
                                        padding = 'valid', # if causal convolution is needed
                                        depth_k=4, depth_v=4,  
                                        num_heads=2, relative_encodings=True)


                        encode_smiles = GlobalMaxPooling1D()(x_smiles)

                        encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
                        encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
                        encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
                        
                        x2 = (encode_protein.shape[1])
                        y2 = (encode_protein.shape[2])
                #uncomment -- 100
                        x_protein = augmented_conv1d(encode_protein, shape = (x2, y2), filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,
                                        strides = 1,
                                        padding = 'valid', # if causal convolution is needed
                                        depth_k=10, depth_v=10,  
                                        num_heads=5, relative_encodings=True)

                        encode_protein = GlobalMaxPooling1D()(x_protein)
                
                
                        encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

                        # Fully connected 
                        FC1 = Dense(1024, activation='relu')(encode_interaction)
                        FC2 = Dropout(0.1)(FC1)
                        FC2 = Dense(1024, activation='relu')(FC2)
                        FC2 = Dropout(0.1)(FC2)
                        FC2 = Dense(512, activation='relu')(FC2)


                        # And add a logistic regression on top
                        predictions = Dense(1, kernel_initializer='glorot_normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'
                        mirrored_strategy = tf.distribute.MirroredStrategy()
                        with mirrored_strategy.scope():
                                interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

                                interactionModel.compile(optimizer='adadelta', loss='mean_squared_error', metrics=[cindex_score]) 
                                print(interactionModel.summary())
                                plot_model(interactionModel, to_file='figures/build_combined_categorical.png')
                                gridmodel = interactionModel

                                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                                gridres = gridmodel.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=batchsz, epochs=epoch, 
                                        validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)),  shuffle=False, callbacks=[es] ) 


                                predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots) ])
                                loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), verbose=0)
                                
                                gridmodel.save("data/models/model.h5")
                                print("Saved model to disk")
                                # list all data in history
                                print(gridres.history.keys())
                                # summarize history for accuracy
                                plotacc(gridres, foldind)
                                # summarize history for loss
                                plotloss(gridres, foldind)
                                # summarize history for scatter
                                plotscatter(foldind, val_Y, predicted_labels)

                        rperf = prfmeasure(val_Y, predicted_labels)
                        rperf = rperf[0]


                        all_predictions[pointer][foldind] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                        all_losses[pointer][foldind]= loss

        pointer +=1

    bestperf = -float('Inf')
    bestpointer = None


    best_param_list = []
    
        ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):
                
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    #print(epoch, batchsz, avgperf)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer +=1

        
    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses


