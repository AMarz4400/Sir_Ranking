# -*- coding: utf-8 -*-

import os
import numpy as np
from torch.utils.data import Dataset
import random


class ReviewData(Dataset):

    def __init__(self, root_path, mode, setup="Default", user=None):
        """
        Modificato per supportare il ranking (BPR) mode.
        Se `ranking=True`, restituisce triple (u, i, j) per la BPR loss.
        """
        self.setup = setup
        if mode == 'Train':
            path = os.path.join(root_path, 'train/')
            print('loading train data')
            self.data = np.load(path + 'Train.npy', allow_pickle=True)
            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(root_path, 'val/')
            print('loading val data')
            self.data = np.load(path + 'Val.npy', allow_pickle=True)
            self.scores = np.load(path + 'Val_Score.npy')
        elif mode == 'Inference':
            path = os.path.join(root_path, 'test/')
            print('loading test data')
            self.data = np.load(path + 'Test.npy', allow_pickle=True)
            user_id = user  # Assumendo che user sia già stato fornito come input alla classe
            all_item_ids = np.unique(self.data[:, 1])
            self.data = [(user_id, item_id) for item_id in all_item_ids]
            print('Ehi, ci sono i dati')
            print(len(self.data))
            self.scores = np.zeros(len(self.data))
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')
            self.data = np.load(path + 'Test.npy', allow_pickle=True)
            self.scores = np.load(path + 'Test_Score.npy')

        if self.setup == "BPR":

            self.len = len(self.data)
            self.mode = mode
            self.positive_items = self._positive_items_dict()

            if self.mode == "Train":
                self.all_items = np.unique(self.data[:, 1])
            else:
                path = os.path.join(root_path, 'train/')
                data = np.load(path + 'Train.npy', allow_pickle=True)
                self.all_items = np.unique(data[:, 1])
                del data

            if mode == "Test":
                path = os.path.join(root_path, 'val/')
                self.data = np.load(path + 'Val.npy', allow_pickle=True)
                self.scores = np.load(path + 'Val_Score.npy')
                self.interacted_val = self._positive_items_dict()
                del self.data, self.scores

            if mode == "Val" or mode == "Test":
                path = os.path.join(root_path, 'train/')
                self.data = np.load(path + 'Train.npy', allow_pickle=True)
                self.scores = np.load(path + 'Train_Score.npy')
                self.interacted_train = self._positive_items_dict()
                del self.data, self.scores
                return

            self.x = self._generate_bpr_triples()
            return

        if self.setup == "Default":
            self.x = list(zip(self.data, self.scores))

    def _positive_items_dict(self):
        """
        Costruisce un dizionario che associa ogni utente agli item con cui ha interagito.
        """
        user_item_dict = {}
        for (user, item), score in zip(self.data, self.scores):

            if score < 4:
                continue

            if user not in user_item_dict:
                user_item_dict[user] = set()
            user_item_dict[user].add(item)
        return user_item_dict

    def _generate_bpr_triples(self):
        all_items = np.unique(self.data[:, 1])  # Ottieni l'insieme di tutti gli item
        triples = []

        for (user, pos_item), score in zip(self.data, self.scores):

            # Trova item negativi (non interagiti dall'utente)
            if user not in self.positive_items:
                continue

            neg_item = self._sample_negative_item(user, all_items)

            if neg_item is not None:
                triples.append((user, pos_item, neg_item))

        return triples

    def _sample_negative_item(self, user, all_items):
        """
        Campiona un item negativo (che l'utente non ha interagito).
        """
        # Item con cui l'utente ha già interagito
        user_items = self.positive_items[user]
        neg_items = list(set(all_items) - user_items)  # Item non interagiti dall'utente

        if len(neg_items) > 0:
            return random.choice(neg_items)
        else:
            return None

    def __getitem__(self, idx):
        if self.setup == "BPR" and self.mode != "Train":
            return None

        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        if self.setup == "BPR" and self.mode != "Train":
            return self.len
        return len(self.x)
