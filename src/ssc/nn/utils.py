from typing import Callable
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm

class Trainer():
    """
    Helper class for model training.
    """
    def __init__(self,
                 train_data: 'list[tuple]',
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: torch.optim.Optimizer,
                 scheduler = None,
                 batch_size: int = 8,
                 valid_steps: int = None,
                 shuffle: bool = False,
                 sort_by_loss: bool = True) -> None:
        """
        Arguments:
        ----------
        train_data : list[tuple]
            List of (X, Y) tuples to train the model on
        
        criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Loss function.

        optimizer : Optimizer
            Loss function optimizer.

        scheduler : Any
            Learning rate scheduler.

        batch_size : int
            Size of each data batch.

        valid_steps : int
            Number of validation batches for each epoch.

        shuffle : bool
            If `True`, data are shuffled at the beginning of each epoch.

        sort_by_loss : 
            If `True`, at the beginning of each epoch data are sorted 
            in descending order with respect of the loss function value.
        """
        self.train_data = train_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.valid_steps = valid_steps
        self.shuffle = shuffle
        self.sort_by_loss = sort_by_loss

        self.train_size = len(self.train_data)
        # Check for valid batch_size
        assert self.batch_size > 0 and self.batch_size < self.train_size
        # Total number of batches
        self.tot_batches = self.train_size // self.batch_size
        # Check for valid valid_steps (at least 1 batch must be used for training)
        if self.valid_steps != None:
            assert self.valid_steps >= 0 and self.valid_steps <= self.tot_batches - 1
        else:
            # Default validation steps
            valid_steps = 0
        # Compute training steps
        self.train_steps = self.tot_batches - self.valid_steps
        # Strat and stop index of validation data
        self.valid_start = self.train_steps * self.batch_size
        self.valid_end = self.valid_start + self.valid_steps * self.batch_size

        self.metrics = []

        # Current batch loss and metric accumulators (train. and valid.)
        self.batch_loss = 0
        self.metric_dict = {}
        self.valid_batch_loss = 0
        self.valid_metric_dict = {}

        # Loss and metric history
        self.loss_history = []
        self.valid_loss_history = []
        self.metric_history = {}
        self.valid_metric_history = {}

        self.epoch_score = []

    def add_metric(self, name: str, metric: Callable, weight: float = 0.0):
        """
        Add the given metric to metric list.
        """
        self.metrics.append((name, metric, weight))
        self.metric_dict[name] = 0
        self.metric_history.update({name : []})
        self.valid_metric_dict["VALID" + name] = 0
        self.valid_metric_history.update({"VALID " + name : []})

    def reset_history(self):
        """
        Reset loss and metrics history.
        """
        self.loss_history = []
        self.valid_loss_history = []
        self.metric_history = {}
        self.valid_metric_history = {}
        for metric_name, _, _ in self.metrics:
            self.metric_history.update({metric_name : []})
            self.valid_metric_history.update({"VALID " + metric_name : []})

    def reset_loss_and_metrics(self):
        """
        Reset current batch loss and metrics.
        """
        self.batch_loss = 0
        self.metric_dict = {}
        for metric_name, _, _ in self.metrics:
            self.metric_dict[metric_name] = 0
    
    def reset_valid_loss_and_metrics(self):
        """
        Reset current validation batch loss and metrics.
        """
        self.valid_batch_loss = 0
        self.valid_metric_dict = {}
        for metric_name, _, _ in self.metrics:
            self.valid_metric_dict["VALID " + metric_name] = 0

    def reset_scores(self):
        """
        Reset epoch scores.
        """
        self.epoch_score = []

    def shuffle_data(self):
        """
        Shuffle training data.
        """
        if self.shuffle and not self.sort_by_loss:
            random.shuffle(self.train_data)
        elif self.sort_by_loss and len(self.epoch_score) > 0:
            sorted_score = np.flip(np.argsort(self.epoch_score))
            self.train_data = [self.train_data[s] for s in sorted_score]

    def optimize(self):
        """
        Make 1 optimization step.
        """
        self.optimizer.zero_grad()
        self.batch_loss.backward()
        self.optimizer.step()
        if self.scheduler != None:
            self.scheduler.step()

    def train_epoch(self, epoch: int, model: nn.Module):
        """
        Train for 1 epoch.
        """
        # Reset epoch scores
        self.reset_scores()
        # Prepare progress bar
        pbar = tqdm(range(self.train_steps + self.valid_steps), desc = "Epoch {}".format(epoch + 1))
        # Training phase
        train_batches = [self.train_data[n : n + self.batch_size] for n in range(self.train_steps)]
        # For each batch
        for batch in train_batches:
            # Reset loss and metrics
            self.reset_loss_and_metrics()
            # For each sample
            for X, Y in batch:
                # Make prediction
                Y_pred = model(X)
                # Compute loss
                loss = self.criterion(Y, Y_pred)
                # Compute and accumulate metrics
                for metric_name, metric, weight in self.metrics:
                    metric_value = metric(Y, Y_pred)
                    # Combine loss and metric
                    if weight != 0:
                        loss += metric_value * weight
                    if torch.is_tensor(metric_value):
                        metric_value = metric_value.item()
                    self.metric_dict[metric_name] += metric_value
                # Accumulate loss
                self.batch_loss += loss
                # Update scores
                self.epoch_score.append(loss.item())

            # Compute the mean loss
            self.batch_loss /= self.batch_size
            # Compute the mean for each metric
            for metric_name, _, _ in self.metrics:
                metric_value = self.metric_dict[metric_name] / self.batch_size
                if torch.is_tensor(metric_value):
                    metric_value = metric_value.item()
                self.metric_dict[metric_name] = metric_value

            # Optimize
            self.optimize()
            # Show batch summary
            summary = {"LOSS": self.batch_loss.item()}
            summary.update(self.metric_dict)
            pbar.update()
            pbar.set_postfix(summary)

        # Update history
        self.loss_history.append(self.batch_loss.item())
        for metric_name, _, _ in self.metrics:
            self.metric_history[metric_name].append(self.metric_dict[metric_name])

        # Validation phase
        model.eval()
        valid_batch = self.train_data[self.valid_start : self.valid_end]
        with torch.no_grad():
            # Reset loss and metrics
            self.reset_valid_loss_and_metrics()
            # For each sample
            for X, Y in valid_batch:
                # Make prediction
                Y_pred = model(X)
                # Compute and accumulate loss
                valid_loss = self.criterion(Y, Y_pred)
                # Compute and accumulate metrics
                for metric_name, metric, weight in self.metrics:
                    metric_name = "VALID " + metric_name
                    metric_value = metric(Y, Y_pred)
                    if weight != 0:
                        valid_loss += metric_value * weight
                    if torch.is_tensor(metric_value):
                        metric_value = metric_value.item()
                    self.valid_metric_dict[metric_name] += metric_value 
                # Accumulate loss
                self.valid_batch_loss += valid_loss.item()
                # Update scores
                self.epoch_score.append(valid_loss.item())

            # Compute the mean loss and add to the history
            self.valid_batch_loss /= len(valid_batch)
            self.valid_loss_history.append(self.valid_batch_loss)
            # Compute the mean for each metric
            for metric_name, _, _ in self.metrics:
                metric_name = "VALID " + metric_name
                metric_value = self.valid_metric_dict[metric_name] / len(valid_batch)
                self.valid_metric_dict[metric_name] = metric_value
                # Update history
                self.valid_metric_history[metric_name].append(metric_value)
            # Show batch summary
            summary = {"LOSS": self.batch_loss.item()}
            summary.update(self.metric_dict)
            summary.update({"VALID LOSS": self.valid_batch_loss})
            summary.update(self.valid_metric_dict)
            pbar.update()
            pbar.set_postfix(summary)
        model.train()       
        pbar.close()

    def train(self,
              model: nn.Module,
              epochs: int):
        """
        Training loop.

        Arguments:
        ----------
        model : nn.Module
            Model to train.

        epochs : int
            Number of epochs.
        """
        # Reset history
        self.reset_history()
        # Reset score list
        self.reset_scores()
        # Main loop
        model.train()
        for epoch in range(epochs):
            # Shuffle data
            self.shuffle_data()
            # Train and validate
            self.train_epoch(epoch=epoch, model=model)