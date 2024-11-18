def recall(no_TP, no_CPS):
    return no_TP/no_CPS

def False_Alarm_Rate(no_pred, no_TP):
    return (no_pred - no_TP)/no_pred

def precision(no_TP, no_pred):
    return no_TP/no_pred

def F2_score(recall, precision):
    return 5 * recall * precision / (recall + 4*precision)
