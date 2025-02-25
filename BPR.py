# -*- encoding: utf-8 -*-
import time
import random

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
now = datetime.now
from dataset import ReviewData
from framework import Model
from models.Losses import *
import models
import config
from sklearn.metrics import roc_auc_score
from main import collate_fn, unpack_input, test, predict


def train_bpr(**kwargs):
    grid = False
    if "grid" in kwargs:
        grid = kwargs["grid"]
        kwargs.pop("grid", None)

    setup = False
    if "setup" in kwargs:
        setup = kwargs["setup"]
        kwargs.pop("setup", None)

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    train_data = ReviewData(opt.data_root, mode="Train", setup="BPR")
    train_data_loader = DataLoader(train_data,
                                   batch_size=opt.batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn(setup="BPR")
                                   )

    val_data = ReviewData(opt.data_root, mode="Val", setup="BPR")

    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # Define the BPR loss function
    bpr_loss_func = BPRLoss()

    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    prev_loss = 1e+10

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...")

        for idx, (user, pos_item, neg_item) in enumerate(train_data_loader):
            # Unpack dei dati BPR (utente, item positivo, item negativo)
            positives, negatives = unpack_input(opt, zip(user, pos_item, neg_item), setup="BPR")

            # Il modello ora riceve i dati unpacked
            pos_scores = model(positives)  # Dati per l'item positivo
            neg_scores = model(negatives)  # Dati per l'item negativo
            del positives, negatives

            # Calcolo della BPR loss
            loss = bpr_loss_func(pos_scores, neg_scores)
            total_loss += loss.item() * len(pos_scores)
            del pos_scores, neg_scores

            # Aggiorna i pesi del modello
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Scheduler per la learning rate
        scheduler.step()

        # Report sulla loss media
        avg_loss = total_loss / len(train_data)
        print(f"\tTraining data: total loss: {total_loss:.4f}, avg loss: {avg_loss:.4f}")

        # Validazione alla fine di ogni epoch
        neg_recall, _ = predict_bpr(model, val_data, opt)

        if neg_recall < min_loss:
            model.save(name=opt.dataset, opt=opt.print_opt)
            print(f"{opt.dataset} | {opt.print_opt}")
            min_loss = neg_recall
            print(f"Model saved with [Recall: {min_loss:.4f}] and [Loss: {total_loss:.4f}]")
        elif neg_recall == min_loss and total_loss < prev_loss:
            model.save(name=opt.dataset, opt=opt.print_opt)
            print(f"{opt.dataset} | {opt.print_opt}")
            print(f"Model saved, same [Recall: {min_loss:.4f}], new [Loss:{total_loss:.4f}]")

        prev_loss = total_loss
        print("*" * 30)

    tmp = "./checkpoints/" + opt.model + "_" + opt.dataset + "_" + opt.print_opt + ".pth"
    model.load(tmp)

# Report finale
    print("----" * 20)
    print(f"Inizio calcolo Recall [{now()}]")
    neg_recall, _ = predict_bpr(model, val_data, opt, U=-1, N=-1)
    print(f"Fine calcolo Recall [{now()}]")
    print("----" * 20)
    print(f"{opt.dataset} {opt.print_opt} best_res: {neg_recall}")
    print("----" * 20)

    if grid:
        return neg_recall, opt.dataset, opt.print_opt


# Funziona sia per validation che per test
def predict_bpr(model, data, opt, k=10, N=200, U=128):
    if N == -1:
        N = float("inf")

    if U == -1:
        U = float("inf")

    recall_k = 0.0
    model.eval()

    with torch.no_grad():

        # Prendo un numero U utenti casuali
        users = list(data.positive_items.keys())
        if len(users) > U:
            users = random.sample(users, U)

        for user in users:
            interacted = []
            positives = []

            for item in data.positive_items[user]:
                positives += [item]

            # Caso Validation: Elimino i positive del train
            if hasattr(data, "interacted_train") and user in data.interacted_train:
                interacted += data.interacted_train[user]

            # Caso Test: Elimino i positive del validation (e train)
            if hasattr(data, "interacted_val") and user in data.interacted_val:
                interacted += data.interacted_val[user]

            # Filtro la lista di tutti gli item positivi
            all_item = [item for item in data.all_items if item not in interacted and item not in positives]

            # Prendo N item casuali negativi
            # NOTA: questa parte può essere migliorata prendendo campioni più simili ai positivi o con altre tecniche
            if len(all_item) > N:
                all_item = random.sample(all_item, N)

            # Aggiungo gli item positivi per la recall
            # Li aggiungo dopo così 1° non vengono filtrati e 2° la lista all_item è composta da item unici
            all_item += positives

            # Variabile temporanea necessaria per effettuare unpack_input
            tmp = [user for _ in range(len(all_item))]
            x = unpack_input(opt, zip(tmp, all_item))
            Y = model(x)
            del tmp, x

            # Creo le coppie (item, score) e riordino il tutto per lo score
            z = list(zip(all_item, Y))
            z = sorted(z, key=lambda pair: pair[1], reverse=True)

            # top_k
            top_k = z[:k]
            del z, Y, all_item

            # recall
            total_relevant = len(data.positive_items[user])
            count = 0
            for item, _ in top_k:
                if item in data.positive_items[user]:
                    count += 1
            recall_k += (count / total_relevant)

    avg_recall = recall_k / len(users)
    model.train()

    # Dato che il "miglior" modello è quello con la recall@k più alta ho messo il '-'
    # Negando si mantiene la logica del codice originale che va in base al valore più basso
    return -recall_k, -avg_recall


def test_bpr(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert (len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")
    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test", setup="BPR")
    print(f"{now()}: test in the test dataset")
    recall, avg_recall, avg_precision, avg_hit_rate, avg_auc  = predict_bpr_with_auc(model, test_data, opt, U=-1, N=-1)
#    test_data = ReviewData(opt.data_root, mode="Test")
#    if len(test_data) == 0:
#        raise ValueError("Il dataset di test è vuoto. Verifica il percorso dei dati o il contenuto del dataset.")
#    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
#    print(f"Numero di batch nel data_loader: {len(test_data_loader)}")
#    print(f"{now()}: test in the test dataset")
    return recall, avg_recall, avg_precision, avg_hit_rate, avg_auc


def predict_bpr_with_auc(model, data, opt, k=10, N=200, U=128):
    if N == -1:
        N = float("inf")
    if U == -1:
        U = float("inf")

    recall_k = 0.0
    precision_k_total = 0.0
    hit_rate_k_total = 0
    auc_total = 0.0  # Aggiungi un accumulatore per l'AUC totale
    model.eval()

    with torch.no_grad():
        users = list(data.positive_items.keys())
        if len(users) > U:
            users = random.sample(users, U)

        for user in users:
            interacted = []
            positives = []
            positives.extend(data.positive_items[user])

            if hasattr(data, "interacted_train") and user in data.interacted_train:
                interacted.extend(data.interacted_train[user])
            if hasattr(data, "interacted_val") and user in data.interacted_val:
                interacted.extend(data.interacted_val[user])

            all_item = [item for item in data.all_items if item not in interacted and item not in positives]
            if len(all_item) > N:
                all_item = random.sample(all_item, N)
            all_item += positives

            if not all_item:  # Verifica che ci siano item validi
                continue

            tmp = [user for _ in range(len(all_item))]
            x = unpack_input(opt, zip(tmp, all_item))
            Y = model(x)
            del tmp, x

            z = list(zip(all_item, Y))
            if not z:  # Assicurati che `z` sia definito e contenga elementi
                continue

            z = sorted(z, key=lambda pair: pair[1], reverse=True)
            top_k = z[:k]
            del z, Y, all_item

            # Calcolo recall@k
            total_relevant = len(data.positive_items[user])
            count = sum(1 for item, _ in top_k if item in data.positive_items[user])
            recall_k += (count / total_relevant)

            # Calcolo precision@k
            precision_k_total += count / k

            # Calcolo hit rate@k
            if count > 0:
                hit_rate_k_total += 1

            # Calcolo AUC per l'utente
            labels = [1 if item in data.positive_items[user] else 0 for item, _ in top_k]
            scores = [score for _, score in top_k]
            if len(set(labels)) > 1:  # Calcola AUC solo se ci sono sia positivi che negativi
                auc_total += roc_auc_score(
                    torch.tensor(labels).cpu().numpy(),
                    torch.tensor(scores).cpu().numpy()
                )

    avg_recall = recall_k / len(users)
    avg_precision = precision_k_total / len(users)
    avg_hit_rate = hit_rate_k_total / len(users)
    avg_auc = auc_total / len(users)  # AUC medio su tutti gli utenti

    model.train()
    return -recall_k, -avg_recall, avg_precision, avg_hit_rate, avg_auc
