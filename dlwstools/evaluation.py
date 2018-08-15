"""
Common utility functions for evaluation.
Licensed under the MIT License (see LICENSE for details)
Written by Hongzhi Li
"""


from sklearn.metrics import *
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def _evaluate(labels,scores,class_names=None, plot_figure=False):
    labels = np.array(labels)
    scores = np.array(scores)

    if len(scores.shape)==3:
        scores = np.average(scores,axis=0)

    if len(labels.shape)==1:
        labels_t = np.zeros_like(scores)
        for i in range(labels.shape[0]):
            labels_t[i,labels[i]] = 1
        labels = labels_t

    labels_sparse = np.argmax(labels,axis=1)
    score_sparse = np.argmax(scores,axis=1)
    
    n_classes = scores.shape[1]
    
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # For each class
    precision = dict()
    recall = dict()
    
    c_precision = dict()
    c_recall = dict()
    c_fscore = dict()    
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            scores[:, i])
        average_precision[i] = average_precision_score(labels[:, i], scores[:, i])

        
        
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(labels.ravel(),
        scores.ravel())
    
    for avg_method in ["macro","micro","weighted"]:
        c_precision[avg_method],c_recall[avg_method],c_fscore[avg_method],_ = precision_recall_fscore_support(labels_sparse, score_sparse, average=avg_method)        
  

    for avg_method in ["macro","micro","weighted"]:
        print('recall score, {0}-averaged over all classes: {1:0.2f}'
              .format(avg_method, c_recall[avg_method]))            
            
    
    for avg_method in ["macro","micro","weighted"]:
        print('precision score, {0}-averaged over all classes: {1:0.2f}'
              .format(avg_method, c_precision[avg_method]))  
        
    for avg_method in ["macro","micro","weighted"]:
        print('F1 score, {0}-averaged over all classes: {1:0.2f}'
              .format(avg_method, c_fscore[avg_method]))          
    


    average_precision["micro"] = average_precision_score(labels, scores,
                                                         average="micro")
    
    average_precision["macro"] = average_precision_score(labels, scores,
                                                         average="macro")
    
    average_precision["weighted"] = average_precision_score(labels, scores,
                                                         average="weighted")    
    
    
    print('Average precision score, macro-averaged over all classes: {0:0.2f}'
          .format(average_precision["macro"]))            
            
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))     
    
    print('Average precision score, weighted-averaged over all classes: {0:0.2f}'
          .format(average_precision["weighted"]))         

        
          
    cm = confusion_matrix(labels_sparse, score_sparse)
    
    print("==========================================")
    print("===========CONFUSION  MATRIX==============")
    print("==========================================")
    
    print(cm)
    
    cm_s = cm.sum(axis=1).repeat(n_classes).reshape(n_classes,n_classes)
    print(cm/cm_s)
        
    from sklearn.metrics import classification_report        
    
    print("==========================================")
    print("========== CLASSIFICATION REPORT =========")
    print("==========================================")
    print(classification_report(labels_sparse, score_sparse, target_names=class_names))
        
    if plot_figure:
        plot_confusion_matrix(cm, class_names,normalize=True)    
        
        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                         color='b')


        from itertools import cycle
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal' ,'c','y','b'])

        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))    
    return c_recall["macro"]
    
def evaluate(labels,scores,class_names=None, plot_figure=False):   
    scores = np.asarray(scores)
    labels = np.asarray(labels) 
    if len(scores.shape)==2:
        _evaluate(labels,scores)
    else:
        overall = []
        for i in range(scores.shape[0]):
            print("=========================================================")
            print("=========Evaluation Report for Model %d==================" % i)
            print("=========================================================")
            main_eval = _evaluate(labels,scores[i,:])
            overall.append(main_eval)
            print("=========================================================")
            print("=========End of Evaluation Report for Model %d===========" % i)
            print("=========================================================\n\n\n")     
            
        print("=========================================================")
        print("=========Evaluation Report for Ensemble Model ===========")
        print("=========================================================")            
        main_eval = _evaluate(labels,scores)
        overall.insert(0,main_eval)
        print("=========================================================")
        print("=========End of Evaluation Report for Ensemble Model======")
        print("=========================================================\n\n\n")       

        
        print("=========================================================")
        print("    Main Evaluation Metric (Macro average of Recall) ")
        print("    (ensemble model, model 1, model 2,......,model n)")
        print("=========================================================")           
        
        print(overall)
        
        
