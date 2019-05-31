import os
from copy import deepcopy
from typing import Dict, Iterable, List

import spacy
import torch
import torchtext
from spacy.symbols import ORTH
from torch.utils.data import DataLoader

from mean_accumulator import MeanAccumulator

from ..utils import DistributedSampler
from .model import RNNModel


class Batch:
    def __init__(self, x, y, hidden):
        self.x = x
        self.y = y
        self.hidden = hidden


ITOS = None  # integer to string
STOI = None  # string to integer


class LanguageModelingTask:
    def __init__(self, device, timer, seed, batch_size):
        self._device = device
        self._timer = timer
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0

        torch.random.manual_seed(self._seed)
        self.text, self.train_loader, self.val_loader = define_dataset(
            device,
            "wikitext2",
            os.path.join(os.getenv("DATA"), "data", "wikitext2"),
            batch_size=self._batch_size,
        )

        global ITOS
        global STOI
        ITOS = self.text.vocab.itos
        STOI = self.text.vocab.stoi

        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self.state = [parameter for parameter in self._model.parameters()]
        self.buffers = [buffer for buffer in self._model.buffers()]
        self.parameter_names = [name for (name, _) in self._model.named_parameters()]
        self._hidden_container = {"hidden": None}

    def train_iterator(self, batch_size: int) -> Iterable[Batch]:
        """Create a dataloader serving `Batch`es from the training dataset.
        Example:
            >>> for batch in task.train_iterator(batch_size=64):
            ...     batch_loss, gradients = task.batchLossAndGradient(batch)
        """
        self._epoch += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1
        self._hidden_container["hidden"] = self._model.init_hidden(batch_size)
        return SplitBatchLoader(
            self.train_loader,
            self._device,
            rank,
            batch_size,
            model=self._model,
            hidden_container=self._hidden_container,
        )

    def batch_loss(self, batch: Batch) -> (float, Dict[str, float]):
        """
        Evaluate the loss on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        Returns:
            - loss function (float)
            - bunch of metrics (dictionary)
        """
        with torch.no_grad():
            with self._timer("batch.forward", float(self._epoch)):
                prediction, hidden = self._model(batch.x, batch.hidden)
                self._hidden_container["hidden"] = hidden
                loss = self._criterion(
                    prediction.view(-1, self._model.ntokens), batch.y.contiguous().view(-1)
                )
            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_prediction(prediction, batch.y)
        return loss.item(), metrics

    def batch_loss_and_gradient(
        self, batch: Batch, rnn_clip=0.4
    ) -> (float, List[torch.Tensor], Dict[str, float]):
        """
        Evaluate the loss and its gradients on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        Returns:
            - function value (float)
            - gradients (list of tensors in the same order as task.state())
            - bunch of metrics (dictionary)
        """
        self._zero_grad()
        with self._timer("batch.forward", float(self._epoch)):
            prediction, hidden = self._model(batch.x, batch.hidden)
            self._hidden_container["hidden"] = hidden
            f = self._criterion(
                prediction.view(-1, self._model.ntokens), batch.y.contiguous().view(-1)
            )
        with self._timer("batch.backward", float(self._epoch)):
            f.backward()
        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_prediction(prediction, batch.y)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), rnn_clip)
        df = [parameter.grad for parameter in self._model.parameters()]
        return f.detach(), df, metrics

    def evaluate_prediction(self, model_output, reference):
        """
        Compute a series of scalar loss values for a predicted batch and references
        """
        with torch.no_grad():
            cross_entropy = self._criterion(
                model_output.view(-1, self._model.ntokens), reference.contiguous().view(-1)
            )
            return {
                "cross_entropy": cross_entropy.detach(),
                "perplexity": torch.exp(cross_entropy).detach(),
            }

    def test(self, state_dict=None) -> float:
        """
        Compute the average loss on the test set.
        The task is completed as soon as the output is below self.target_test_loss.
        If the model has batch normalization or dropout, this will run in eval mode.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1

        self._hidden_container["hidden"] = self._model.init_hidden(self._batch_size)
        test_loader = SplitBatchLoader(
            self.val_loader,
            self._device,
            rank,
            batch_size=self._batch_size,
            model=self._model,
            hidden_container=self._hidden_container,
        )

        if state_dict:
            test_model = self._create_test_model(state_dict)
        else:
            test_model = self._model
            test_model.eval()

        mean_metrics = MeanAccumulator()

        for batch in test_loader:
            with torch.no_grad():
                prediction, hidden = self._model(batch.x, batch.hidden)
                self._hidden_container["hidden"] = hidden
                metrics = self.evaluate_prediction(prediction, batch.y)
            mean_metrics.add(metrics)
        mean_metrics.reduce()  # Collect over workers

        test_model.train()
        return mean_metrics.value()

    def state_dict(self):
        """Dictionary containing the model state (buffers + tensors)"""
        return self._model.state_dict()

    def _create_model(self):
        """Create a PyTorch module for the model"""
        torch.random.manual_seed(self._seed)
        model = define_model(self.text)
        model.to(self._device)
        model.train()
        return model

    def _create_test_model(self, state_dict):
        test_model = deepcopy(self._model)
        test_model.load_state_dict(state_dict)
        test_model.eval()
        return test_model

    def _zero_grad(self):
        self._model.zero_grad()


class SplitBatchLoader:
    """
    Utility that transforms a DataLoader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device, rank, batch_size, model, hidden_container):
        self._dataloader = dataloader
        self._device = device
        self._rank = rank
        self._batch_size = batch_size
        self._model = model
        self._hidden_container = hidden_container

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for i, batch in enumerate(self._dataloader):
            # if i == 0:
            #     print("Data signature", batch.text.view(-1)[0:5].numpy())
            x = batch.text[:, self._rank * self._batch_size : (self._rank + 1) * self._batch_size]
            y = batch.target[:, self._rank * self._batch_size : (self._rank + 1) * self._batch_size]
            hidden = self._model.repackage_hidden(self._hidden_container["hidden"])
            yield Batch(x, y, hidden)


def define_dataset(
    device,
    dataset_name,
    dataset_path,
    batch_size,
    rnn_use_pretrained_emb=False,
    rnn_n_hidden=650,
    reshuffle_per_epoch=True,
    rnn_bptt_len=30,
):
    # create dataset.
    TEXT, train, valid, test = _get_dataset(dataset_name, dataset_path)

    n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1

    # Build vocb.
    # we can use some precomputed word embeddings,
    # e.g., GloVe vectors with 100, 200, and 300.
    if rnn_use_pretrained_emb:
        try:
            vectors = "glove.6B.{}d".format(rnn_n_hidden)
            vectors_cache = os.path.join(dataset_path, ".vector_cache")
        except:
            vectors, vectors_cache = None, None
    else:
        vectors, vectors_cache = None, None
    TEXT.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache)

    # Partition training data.
    train_loader, _ = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size * n_workers,
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )
    _, val_loader = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size * n_workers,
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )

    # get some stat.
    return TEXT, train_loader, val_loader


def define_model(TEXT, rnn_n_hidden=650, rnn_n_layers=3, rnn_tie_weights=True, drop_rate=0.4):
    # get embdding size and num_tokens.
    weight_matrix = TEXT.vocab.vectors

    if weight_matrix is not None:
        n_tokens, emb_size = weight_matrix.size(0), weight_matrix.size(1)
    else:
        n_tokens, emb_size = len(TEXT.vocab), rnn_n_hidden

    # create model.
    model = RNNModel(
        rnn_type="LSTM",
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
    )

    # init the model.
    if weight_matrix is not None:
        model.encoder.weight.data.copy_(weight_matrix)

    return model


def _get_text():
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok)
    return TEXT


def _get_dataset(name, datasets_path):
    TEXT = _get_text()

    # Load and split data.
    if "wikitext2" in name:
        train, valid, test = torchtext.datasets.WikiText2.splits(TEXT, root=datasets_path)
    elif "ptb" in name:
        train, valid, test = torchtext.datasets.PennTreebank.splits(TEXT, root=datasets_path)
    return TEXT, train, valid, test
