########################
#Delete .cuda() or .to() calls
##########################
# before lightning
def forward(self, x):
    x = x.cuda(0)
    layer_1.cuda(0)
    x_hat = layer_1(x)


# after lightning
def forward(self, x):
    x_hat = layer_1(x)


########################
# Init tensors using Tensor.to and register_buffer
# When you need to create a new tensor, use Tensor.to. 
# This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.
##########################

# before lightning
def forward(self, x):
    z = torch.Tensor(2, 3)
    z = z.cuda(0)


# with lightning
def forward(self, x):
    z = torch.Tensor(2, 3)
    z = z.to(x)
#The LightningModule knows what device it is on. 
# You can access the reference via self.device. 
# Sometimes it is necessary to store tensors as module attributes. 
# However, if they are not parameters they will remain on the CPU even if the module gets moved to a new device. 
# To prevent that and remain device agnostic, 
# register the tensor as a buffer in your modules’ __init__ method with register_buffer().

class LitModel(LightningModule):
    def __init__(self):
        ...
        self.register_buffer("sigma", torch.eye(3))
        # you can now access self.sigma anywhere in your module

##########################
#Remove samplers
#DistributedSampler is automatically handled by Lightning.
##########################
#See use_distributed_sampler for more information.




##########################
#
#Synchronize validation and test logging
#
##########################
#When running in distributed mode, we have to ensure that the validation and test step logging calls are synchronized across processes. This is done by adding sync_dist=True to all self.log calls in the validation and test step. This ensures that each GPU worker has the same behaviour when tracking model checkpoints, which is important for later downstream tasks such as testing the best checkpoint across all workers. The sync_dist option can also be used in logging calls during the step methods, but be aware that this can lead to significant communication overhead and slow down your training.

#Note if you use any built in metrics or custom metrics that use TorchMetrics, these do not need to be updated and are automatically handled for you.

def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
    self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)


def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
    self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

# It is possible to perform some computation manually and log the reduced result on rank 0 as follows:

def __init__(self):
    super().__init__()
    self.outputs = []


def test_step(self, batch, batch_idx):
    x, y = batch
    tensors = self(x)
    self.outputs.append(tensors)
    return tensors


def on_test_epoch_end(self):
    mean = torch.mean(self.all_gather(self.outputs))
    self.outputs.clear()  # free memory

    # When logging only on rank 0, don't forget to add
    # `rank_zero_only=True` to avoid deadlocks on synchronization.
    # caveat: monitoring this is unimplemented. see https://github.com/Lightning-AI/lightning/issues/15852
    if self.trainer.is_global_zero:
        self.log("my_reduced_metric", mean, rank_zero_only=True)


##########################
#
#Make models pickleable
#
##########################
import pickle
pickle.dump(some_object)


##########################
#
#Train on GPUs
#
##########################
# The Trainer will run on all available GPUs by default. Make sure you’re running on a machine with at least one GPU. 
# There’s no need to specify any NVIDIA flags as Lightning will do it for you.
# run on as many GPUs as available by default
trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
# equivalent to
trainer = Trainer()

# run on one GPU
trainer = Trainer(accelerator="gpu", devices=1)
# run on multiple GPUs
trainer = Trainer(accelerator="gpu", devices=8)
# choose the number of devices automatically
trainer = Trainer(accelerator="gpu", devices="auto")

# DEFAULT (int) specifies how many GPUs to use per node
Trainer(accelerator="gpu", devices=k)

# Above is equivalent to
Trainer(accelerator="gpu", devices=list(range(k)))

# Specify which GPUs to use (don't use when running on cluster)
Trainer(accelerator="gpu", devices=[0, 1])

# Equivalent using a string
Trainer(accelerator="gpu", devices="0, 1")

# To use all available GPUs put -1 or '-1'
# equivalent to list(range(torch.cuda.device_count()))
Trainer(accelerator="gpu", devices=-1)



##########################
#
# datamodule !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# see https://lightning.ai/docs/pytorch/stable/data/datamodule.html
# for data agnost
##########################
#A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data:
#A datamodule encapsulates the five steps involved in data processing in PyTorch:

#1.Download / tokenize / process.
#2.Clean and (maybe) save to disk.
#3.Load inside Dataset.
#4.Apply transforms (rotate, tokenize, etc…).
#5.Wrap inside a DataLoader.

#This class can then be shared and used anywhere:

model = LitClassifier()
trainer = Trainer()

imagenet = ImagenetDataModule()
trainer.fit(model, datamodule=imagenet)

cifar10 = CIFAR10DataModule()
trainer.fit(model, datamodule=cifar10)



############################################
# Save checkpoints by condition
# see https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html
################################################
# 3. Init ModelCheckpoint callback, monitoring "val_loss"
from lightning.pytorch.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    dirpath="my/path/",
    filename="sample-mnist-{epoch:02d}-{global_step}",
)
# 4. Add your callback to the callbacks list
trainer = Trainer(callbacks=[checkpoint_callback])



############################################
# Print things while traning
# see https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html
################################################
trainer = Trainer(log_every_n_steps=10)


##############################################
# Precision
##############################################
# you can also get a ~3x speed improvement. 
# Half precision can sometimes lead to unstable training.
Trainer(precision='16-mixed')
Trainer(precision="32")
Trainer(precision="64")



##################################################
#
# Accumulate Gradients !!!!!!!!!!!!!!!!!!!
#
#####################################################
#Accumulated gradients run K small batches of size N before doing a backward pass. 
# The effect is a large effective batch size of size KxN,
# Accumulate gradients for 7 batches
trainer = Trainer(accumulate_grad_batches=7)


##################################################
#
# Gradient Clipping
# see https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
#####################################################
# clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
trainer = Trainer(gradient_clip_val=0.5)

# clip gradients' maximum magnitude to <=0.5
trainer = Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")



##################################################
#https://www.bilibili.com/video/av759964881/?vd_source=4c11776b1786335addbff5fb554b0229
#Stochastic Weight Averaging
#
#
#####################################################
trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])


##################################################
#
# Max Batch Size Finder
#
#####################################################

from lightning.pytorch.tuner import Tuner

# Create a tuner for the trainer
trainer = Trainer(...)
tuner = Tuner(trainer)

# Auto-scale batch size by growing it exponentially (default)
tuner.scale_batch_size(model, mode="power")

# Auto-scale batch size with binary search
tuner.scale_batch_size(model, mode="binsearch")

# Fit as normal with 
trainer.fit(model)

#######################################################33
#
#Sharing Datasets Across Process Boundaries
#
######################################################
# each replica consums cpu memorty to store data
# use shard data
model = Model(...)
datamodule = MNISTDataModule("data/MNIST")

trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_spawn")
trainer.fit(model, datamodule)

##################################
#
# Setup the cluster
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html
#####################################
'''
Setup the cluster
This guide shows how to run a training job on a general purpose cluster. We recommend beginners to try this method first because it requires the least amount of configuration and changes to the code. To setup a multi-node computing cluster you need:

Multiple computers with PyTorch Lightning installed

A network connectivity between them with firewall rules that allow traffic flow on a specified MASTER_PORT.

Defined environment variables on each node required for the PyTorch Lightning multi-node distributed training

PyTorch Lightning follows the design of PyTorch distributed communication package. and requires the following environment variables to be defined on each node:

MASTER_PORT - required; has to be a free port on machine with NODE_RANK 0

MASTER_ADDR - required (except for NODE_RANK 0); address of NODE_RANK 0 node

WORLD_SIZE - required; the total number of GPUs/processes that you will use

NODE_RANK - required; id of the node in the cluster
'''

#not need torch.distributed.run again!!!

# train on 32 GPUs across 4 nodes
trainer = Trainer(accelerator="gpu", devices=8, num_nodes=4, strategy="ddp")

'''
Submit a job to the cluster
To submit a training job to the cluster you need to run the same training script on each node of the cluster. This means that you need to:

Copy all third-party libraries to each node (usually means - distribute requirements.txt file and install it).

Copy all your import dependencies and the script itself to each node.

Run the script on each node.
'''

'''
DEBUG
'''
NCCL_DEBUG=INFO python train.py ...













