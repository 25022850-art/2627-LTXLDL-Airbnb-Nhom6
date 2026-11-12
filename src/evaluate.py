import numpy as np
import pandas as pd

def eval_sentiment(preds, nhan_tay):
    TP_positive = (preds["general_sentiment"]=="Positive") & (nhan_tay["general_sentiment"]=="Positive")
    FP_positive = (preds["general_sentiment"]=="Positive") & (nhan_tay["general_sentiment"]!="Positive")
    FN_positive = (preds["general_sentiment"]!="Positive") & (nhan_tay["general_sentiment"]=="Positive")
    
    TP_mixed = (preds["general_sentiment"]=="Mixed") & (nhan_tay["general_sentiment"]=="Mixed")
    FP_mixed = (preds["general_sentiment"]=="Mixed") & (nhan_tay["general_sentiment"]!="Mixed")
    FN_mixed = (preds["general_sentiment"]!="Mixed") & (nhan_tay["general_sentiment"]=="Mixed")
    
    TP_negative = (preds["general_sentiment"]=="Negative") & (nhan_tay["general_sentiment"]=="Negative")
    FP_negative = (preds["general_sentiment"]=="Negative") & (nhan_tay["general_sentiment"]!="Negative")
    FN_negative = (preds["general_sentiment"]!="Negative") & (nhan_tay["general_sentiment"]=="Negative")

    if (2 * TP_positive.sum() + FP_positive.sum() + FN_positive.sum()) == 0:
        F1_positive = None
    else:
        F1_positive = 2 * TP_positive.sum() / (2 * TP_positive.sum() + FP_positive.sum() + FN_positive.sum())

    if (2 * TP_mixed.sum() + FP_mixed.sum() + FN_mixed.sum()) == 0:
        F1_mixed = None
    else:
        F1_mixed = 2 * TP_mixed.sum() / (2 * TP_mixed.sum() + FP_mixed.sum() + FN_mixed.sum())

    if (2 * TP_negative.sum() + FP_negative.sum() + FN_negative.sum()) == 0:
        F1_negative = None
    else:
        F1_negative = 2 * TP_negative.sum() / (2 * TP_negative.sum() + FP_negative.sum() + FN_negative.sum())

    if F1_positive is None or F1_mixed is None or F1_negative is None:
        return None
    return (F1_positive + F1_mixed + F1_negative) / 3

def eval_accessibility(preds, nhan_tay):
    TP_highly_accessible = (preds["accessibility"]=="Highly accessible") & (nhan_tay["accessibility"]=="Highly accessible")
    FP_highly_accessible = (preds["accessibility"]=="Highly accessible") & (nhan_tay["accessibility"]!="Highly accessible")
    FN_highly_accessible = (preds["accessibility"]!="Highly accessible") & (nhan_tay["accessibility"]=="Highly accessible")

    TP_poor_access = (preds["accessibility"]=="Poor access") & (nhan_tay["accessibility"]=="Poor access")
    FP_poor_access = (preds["accessibility"]=="Poor access") & (nhan_tay["accessibility"]!="Poor access")
    FN_poor_access = (preds["accessibility"]!="Poor access") & (nhan_tay["accessibility"]=="Poor access")

    TP_neutral = (preds["accessibility"]=="Neutral") & (nhan_tay["accessibility"]=="Neutral")
    FP_neutral = (preds["accessibility"]=="Neutral") & (nhan_tay["accessibility"]!="Neutral")
    FN_neutral = (preds["accessibility"]!="Neutral") & (nhan_tay["accessibility"]=="Neutral")

    TP_not_mentioned = (preds["accessibility"]=="Not mentioned") & (nhan_tay["accessibility"]=="Not mentioned")
    FP_not_mentioned = (preds["accessibility"]=="Not mentioned") & (nhan_tay["accessibility"]!="Not mentioned")
    FN_not_mentioned = (preds["accessibility"]!="Not mentioned") & (nhan_tay["accessibility"]=="Not mentioned")

    if (2 * TP_highly_accessible.sum() + FP_highly_accessible.sum() + FN_highly_accessible.sum()) == 0:
        F1_highly_accessible = None
    else:
        F1_highly_accessible = 2 * TP_highly_accessible.sum() / (2 * TP_highly_accessible.sum() + FP_highly_accessible.sum() + FN_highly_accessible.sum())

    if (2 * TP_poor_access.sum() + FP_poor_access.sum() + FN_poor_access.sum()) == 0:
        F1_poor_access = None
    else:
        F1_poor_access = 2 * TP_poor_access.sum() / (2 * TP_poor_access.sum() + FP_poor_access.sum() + FN_poor_access.sum())

    if (2 * TP_neutral.sum() + FP_neutral.sum() + FN_neutral.sum()) == 0:
        F1_neutral = None
    else:
        F1_neutral = 2 * TP_neutral.sum() / (2 * TP_neutral.sum() + FP_neutral.sum() + FN_neutral.sum())

    if (2 * TP_not_mentioned.sum() + FP_not_mentioned.sum() + FN_not_mentioned.sum()) == 0:
        F1_not_mentioned = None
    else:
        F1_not_mentioned = 2 * TP_not_mentioned.sum() / (2 * TP_not_mentioned.sum() + FP_not_mentioned.sum() + FN_not_mentioned.sum())

    if F1_highly_accessible is None or F1_poor_access is None or F1_neutral is None or F1_not_mentioned is None:
        return None
    return (F1_highly_accessible + F1_poor_access + F1_neutral + F1_not_mentioned) / 4


def eval_noise(preds, nhan_tay):
    TP_quiet = (preds["noise"]=="Quiet") & (nhan_tay["noise"]=="Quiet")
    FP_quiet = (preds["noise"]=="Quiet") & (nhan_tay["noise"]!="Quiet")
    FN_quiet = (preds["noise"]!="Quiet") & (nhan_tay["noise"]=="Quiet")

    TP_noisy = (preds["noise"]=="Noisy") & (nhan_tay["noise"]=="Noisy")
    FP_noisy = (preds["noise"]=="Noisy") & (nhan_tay["noise"]!="Noisy")
    FN_noisy = (preds["noise"]!="Noisy") & (nhan_tay["noise"]=="Noisy")

    TP_acceptable = (preds["noise"]=="Acceptable") & (nhan_tay["noise"]=="Acceptable")
    FP_acceptable = (preds["noise"]=="Acceptable") & (nhan_tay["noise"]!="Acceptable")
    FN_acceptable = (preds["noise"]!="Acceptable") & (nhan_tay["noise"]=="Acceptable")

    TP_not_mentioned = (preds["noise"]=="Not mentioned") & (nhan_tay["noise"]=="Not mentioned")
    FP_not_mentioned = (preds["noise"]=="Not mentioned") & (nhan_tay["noise"]!="Not mentioned")
    FN_not_mentioned = (preds["noise"]!="Not mentioned") & (nhan_tay["noise"]=="Not mentioned")

    if (2 * TP_quiet.sum() + FP_quiet.sum() + FN_quiet.sum()) == 0:
        F1_quiet = None
    else:
        F1_quiet = 2 * TP_quiet.sum() / (2 * TP_quiet.sum() + FP_quiet.sum() + FN_quiet.sum())

    if (2 * TP_noisy.sum() + FP_noisy.sum() + FN_noisy.sum()) == 0:
        F1_noisy = None
    else:
        F1_noisy = 2 * TP_noisy.sum() / (2 * TP_noisy.sum() + FP_noisy.sum() + FN_noisy.sum())

    if (2 * TP_acceptable.sum() + FP_acceptable.sum() + FN_acceptable.sum()) == 0:
        F1_acceptable = None
    else:
        F1_acceptable = 2 * TP_acceptable.sum() / (2 * TP_acceptable.sum() + FP_acceptable.sum() + FN_acceptable.sum())

    if (2 * TP_not_mentioned.sum() + FP_not_mentioned.sum() + FN_not_mentioned.sum()) == 0:
        F1_not_mentioned = None
    else:
        F1_not_mentioned = 2 * TP_not_mentioned.sum() / (2 * TP_not_mentioned.sum() + FP_not_mentioned.sum() + FN_not_mentioned.sum())

    if F1_quiet is None or F1_noisy is None or F1_acceptable is None or F1_not_mentioned is None:
        return None
    return (F1_quiet + F1_noisy + F1_acceptable + F1_not_mentioned) / 4

def evaluate(preds, nhan_tay):
    f1_sentiment = eval_sentiment(preds, nhan_tay)
    f1_accessibility = eval_accessibility(preds, nhan_tay)
    f1_noise = eval_noise(preds, nhan_tay)
    return {
        "f1_sentiment": f1_sentiment,
        "f1_accessibility": f1_accessibility,
        "f1_noise": f1_noise
    }