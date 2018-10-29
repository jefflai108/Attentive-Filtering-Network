import numpy as np
import logging
from timeit import default_timer as timer
from scipy.optimize import fmin_l_bfgs_b, basinhopping
import torch
import torch.nn.functional as F
from v1_metrics import compute_eer, compute_confuse
import data_reader.adv_kaldi_io as ako

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def validation(args, model, device, train_loader, val_loader, val_scp, val_utt2label):
    logger.info("Starting Validation")
    train_loss, _, train_correct = compute_loss(model, device, train_loader)
    val_loss, val_scores, val_correct = compute_loss(model, device, val_loader)
    predictions, labels =  utt_scores(val_scores, val_scp, val_utt2label)
    val_eer, threshold  = best_eer(labels, predictions)

    logger.info('\n===> Training set: Average loss: {:.4f}\tAccuracy: {}/{} ({:.0f}%)\n'.format(
                train_loss, train_correct, len(train_loader.dataset),
                100. * train_correct / len(train_loader.dataset)
                ))
    logger.info('===> Validation set: Average loss: {:.4f}\tEER: {:.4f}\tThreshold: {}\n'.format(
                val_loss, val_eer, threshold))
    return val_loss, val_eer, threshold

def best_eer(true_labels, predictions):

    def f_neg(threshold):
        ## Scipy tries to minimize the function
        return compute_eer(true_labels, predictions >= threshold)

    # Initialization of best threshold search
    thr_0 = [0.20] * 1 # binary class
    constraints = [(0.,1.)] * 1 # binary class
    def bounds(**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0))
        return tmax and tmin

    # Search using L-BFGS-B, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {"method": "L-BFGS-B",
                        "bounds":constraints,
                        "options":{
                            "eps": 0.05
                            }
                       }

    # We combine L-BFGS-B with Basinhopping for stochastic search with random steps
    logger.info("===> Searching optimal threshold for each label")
    start_time = timer()

    opt_output = basinhopping(f_neg, thr_0,
                                stepsize = 0.1,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=10,
                                accept_test=bounds)

    end_time = timer()
    logger.info("===> Optimal threshold for each label:\n{}".format(opt_output.x))
    logger.info("Threshold found in: %s seconds" % (end_time - start_time))

    score = opt_output.fun
    return score, opt_output.x

def utt_scores(scores, scp, utt2label):
    """return predictions and labels per utterance
    """
    utt2len   = ako.read_key_len(scp)
    utt2label = ako.read_key_label(utt2label)
    key_list  = ako.read_all_key(scp)

    preds, labels = [], []
    idx = 0
    for key in key_list:
        frames_per_utt = utt2len[key]
        avg_scores = np.average(scores[idx:idx+frames_per_utt])
        idx = idx + frames_per_utt
        preds.append(avg_scores)
        labels.append(utt2label[key])

    return np.array(preds), np.array(labels)

def compute_loss(model, device, data_loader, threshold=0.5):
    model.eval()
    loss = 0
    correct = 0
    scores  = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1,1).float()
            #output, hidden = model(data, None)
            output = model(data)
            loss += F.binary_cross_entropy(output, target, size_average=False)
            pred = output > 0.5
            correct += pred.byte().eq(target.byte()).sum().item() # not really meaningful

            scores.append(output.data.cpu().numpy())

    loss /= len(data_loader.dataset) # average loss
    scores = np.vstack(scores) # scores per frame

    return loss, scores, correct

