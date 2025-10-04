import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from tqdm.autonotebook import tqdm

class BaseINR(nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(BaseINR, self).__init__()

        self.model = None
        self.device = device
        self.optimizer = None
        self.loss_function = nn.MSELoss().to(self.device)



    def compile(self, optimizer_name="adam", learning_rate=1e-4, scheduler=None,):
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer not supported")

        if scheduler is not None:
            self.scheduler = scheduler
        else:   
            self.scheduler = None

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function.to(self.device)
        
    def fit(self, input_vectors, signal, epochs=1000, report_metrics=None, disable_tqdm=False):
        # self.metrics_dict_train.reset()
        input_vectors = input_vectors.to(self.device)
        signal = signal.to(self.device)
        pbar = tqdm(range(epochs)) if not disable_tqdm else range(epochs)

        compute_metrics_for_training = False

        if report_metrics is not None:
            compute_metrics_for_training = True
            # metric_loggers = self.initialize_metric_loggers(self.metrics_dict_train, report_metrics)
        for epoch in pbar:
            self.optimizer.zero_grad()
            model_output = self(input_vectors)
            loss = self.loss_function(model_output, signal)
            loss.backward()
            self.optimizer.step()
            # if compute_metrics_for_training:
                # metrics_string = self.get_train_metrics(model_output, signal, metric_loggers)
            if not disable_tqdm:
                metrics_string = ''
                pbar.set_description(f"Epoch: {epoch}/{epochs}. Loss: {loss.item():.6f} {metrics_string if report_metrics is not None else ''}")
            if self.scheduler is not None:
                self.scheduler.step()

    def predict(self, input_vectors):
        with torch.no_grad():
            input_vectors = input_vectors.to(self.device)
            return self(input_vectors).cpu().numpy()
    
    def predict_allow_gradients(self, input_vectors):
        input_vectors = input_vectors.to(self.device)
        return self(input_vectors)

    
    def get_weights(self):
        return self.state_dict()

    def copy_from_model(self, model):
        self.load_state_dict(deepcopy({ k:v.detach().clone() for k,v in model.state_dict().items()} ))