import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.utils.data as data

from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--devices", type=int, default=1)

# Hyperparameters for the model
parser.add_argument("--layer_1_dim", type=int, default=128)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        #To track a metric, simply use the self.log method available inside the LightningModule
        self.log("test_loss", test_loss, prog_bar=True)#To view metrics in the commandline progress bar, set the prog_bar argument to True.
        # self.log_dict({"loss": loss, "acc": acc, "metric_n": metric_n})
        #tensorboard --logdir=lightning_logs/
        self.log(..., reduce_fx="mean")#log value of batch,not averaged on epoch
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     # enable Monte Carlo Dropout
    #     self.dropout.train()

    #     # take average of `self.mc_iteration` iterations
    #     pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
    #     pred = torch.vstack(pred).mean(dim=0)
    #     return pred
    # #predictions = trainer.predict(model, data_loader)

# Load data sets
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

from lightning.pytorch.profilers import AdvancedProfiler
profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")#or simple

from lightning.pytorch.callbacks import DeviceStatsMonitor

# train model
trainer = pl.Trainer(max_epochs=5,
                     devices=args.devices,
                     default_root_dir="./",
                    #  fast_dev_run=7, for debug.The fast_dev_run argument in the trainer runs 7 batch of training, validation, test and prediction data through your trainer to see if there are any bugs:
                     limit_train_batches=0.05, #use 5%
                     limit_val_batches=0.01,#use 1%
                     enable_model_summary=False,#not print model summary
                     profiler=profiler,
                     callbacks=[DeviceStatsMonitor()]
                     )
# trainer = Trainer(profiler=profiler)

# print model patameter summary
# from lightning.pytorch.utilities.model_summary import ModelSummary
# summary = ModelSummary(model, max_depth=-1)
# print(summary)

trainer.fit(model=autoencoder, 
            train_dataloaders=DataLoader(train_set, num_workers=2), 
            val_dataloaders=DataLoader(valid_set, num_workers=2))

# test the model
trainer.test(autoencoder, dataloaders=DataLoader(test_set, num_workers=2))


# trainer.fit
# autoencoder = LitAutoEncoder(Encoder(), Decoder())
# optimizer = autoencoder.configure_optimizers()

# for batch_idx, batch in enumerate(train_loader):
#     loss = autoencoder.training_step(batch, batch_idx)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()



#eval
# model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# # disable randomness, dropout, etc...
# model.eval()

# # predict with the model
# y_hat = model(x)

#checkpoint
# checkpoint = torch.load(CKPT_PATH)
# encoder_weights = checkpoint["encoder"]
# decoder_weights = checkpoint["decoder"]
# print(checkpoint["hyper_parameters"])
# # automatically restores model, epoch, step, LR schedulers, etc...
# trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")


#trainer callback
#val_accuracy is the log value name
# early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
# trainer = Trainer(callbacks=[early_stop_callback])


#########################
# Enable distributed inference
# By using the predict step in Lightning you get free distributed inference using BasePredictionWriter.
################################
# import torch
# from lightning.pytorch.callbacks import BasePredictionWriter


# class CustomWriter(BasePredictionWriter):
#     def __init__(self, output_dir, write_interval):
#         super().__init__(write_interval)
#         self.output_dir = output_dir

#     def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
#         # this will create N (num processes) files in `output_dir` each containing
#         # the predictions of it's respective rank
#         torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

#         # optionally, you can also save `batch_indices` to get the information about the data index
#         # from your prediction data
#         torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


# # or you can set `writer_interval="batch"` and override `write_on_batch_end` to save
# # predictions at batch level
# pred_writer = CustomWriter(output_dir="pred_path", write_interval="epoch")
# trainer = Trainer(accelerator="gpu", strategy="ddp", devices=8, callbacks=[pred_writer])
# model = BoringModel()
# trainer.predict(model, return_predictions=False)