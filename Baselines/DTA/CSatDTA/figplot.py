import matplotlib.pyplot as plt
import time
#plot diffrent figures after evaluation
def plotloss(gridres, foldind):
    figname = "b"+str( foldind) + "_" + str(time.time()) 
    plt.plot(gridres.history['loss'])
    plt.plot(gridres.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.savefig("figures/"+figname +"_loss.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()
    
def plotacc(gridres, foldind):
    figname = "b"+str( foldind) + "_" + str(time.time()) 
    plt.plot(gridres.history['cindex_score'])
    plt.plot(gridres.history['val_cindex_score'])
    plt.title('model accuracy')
    plt.ylabel('cindex_score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.savefig("figures/"+figname +"_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()
    
def plotscatter(foldind, val_Y, predicted_labels):
        ## PLOT Scatter
    figname = "b"+str( foldind) + "_" + str(time.time())
    plt.scatter(val_Y, predicted_labels, c='crimson')
    plt.yscale('linear')
    plt.xscale('linear')

    p1 = max(max(predicted_labels), max(val_Y))
    p2 = min(min(predicted_labels), min(val_Y))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.savefig("figures/"+figname + "_scatter.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()