#########################
#
# Callback
# like mmdetection hook
###########################
from lightning.pytorch.callbacks import Callback


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


trainer = Trainer(callbacks=[MyPrintingCallback()])


#########################
#
# TQDMProgressBar
# https://lightning.ai/docs/pytorch/stable/common/progress_bar.html
###########################
#You can update refresh_rate (rate (number of batches) at which the progress bar get updated) for TQDMProgressBar by:
from lightning.pytorch.callbacks import TQDMProgressBar

trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=10)])


############################
#
# RichProgressBar
#
#################################33
#Rich is a Python library for rich text and beautiful formatting in the terminal. 
# To use the RichProgressBar as your progress bar, first install the package:
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

# create your own theme!
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    )
)

trainer = Trainer(callbacks=progress_bar)

# If you wish for a new progress bar to be displayed at the end of every epoch, you should enable RichProgressBar.leave by passing True
from lightning.pytorch.callbacks import RichProgressBar

trainer = Trainer(callbacks=[RichProgressBar(leave=True)])

# Disable progress bar
trainer = Trainer(enable_progress_bar=False)



##################################
#
# Manual Optimization
# https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html
##########################################
from lightning.pytorch import LightningModule


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()


#Access your Own Optimizer
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        ...

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
        # To access it, do the following.
        # However, it won't work on TPU, AMP, etc...
        optimizer = optimizer.optimizer
        ...

import torch
from torch import Tensor
from lightning.pytorch import LightningModule


class SimpleGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def sample_z(self, n) -> Tensor:
        sample = self._Z.sample((n,))
        return sample

    def sample_G(self, n) -> Tensor:
        z = self.sample_z(n)
        return self.G(z)

    def training_step(self, batch, batch_idx):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()

        X, _ = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt
    

#############################
#
# Modify a checkpoint anywhere
#
################################
#When you need to change the components of a checkpoint before saving or loading, 
# use the on_save_checkpoint() and on_load_checkpoint() of your LightningModule.
class LitModel(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint):
        checkpoint["something_cool_i_want_to_save"] = my_cool_pickable_object

    def on_load_checkpoint(self, checkpoint):
        my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]

# OR
class LitCallback(pl.Callback):
    def on_save_checkpoint(self, checkpoint):
        checkpoint["something_cool_i_want_to_save"] = my_cool_pickable_object

    def on_load_checkpoint(self, checkpoint):
        my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]


###################################33
#
# DistributedDataParallel (DDP)
#
################################
from lightning.pytorch.strategies import DDPStrategy

# Explicitly specify the process group backend if you choose to
ddp = DDPStrategy(process_group_backend="nccl")
# train on 32 GPUs (4 nodes)
trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp", num_nodes=4)
# example for 3 GPUs DDP
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc


# note that some of the extreme memory saving configurations will affect the speed of training. 
# This Speed/Memory trade-off in most cases can be adjusted.
# https://www.youtube.com/embed/w_CKzh5C1K4
#
#https://kgithub.com/SeanNaren/minGPT
#
'''
Overall:

When fine-tuning a model, use advanced memory efficient strategies such as Fully Sharded Training, 
DeepSpeed ZeRO Stage 3 or DeepSpeed ZeRO Stage 3 Offload, 
allowing you to fine-tune larger models if you are limited on compute

When pre-training a model, use simpler optimizations such as DeepSpeed ZeRO Stage 2, 
scaling the number of GPUs to reach larger parameter sizes

For both fine-tuning and pre-training, 
use DeepSpeed Activation Checkpointing as the throughput degradation is not significant
'''

'''
For example when using 128 GPUs, you can pre-train large 10 to 20 Billion parameter models 
using DeepSpeed ZeRO Stage 2 without having to take a performance hit with more advanced optimized multi-gpu strategy.

But for fine-tuning a model, you can reach 10 to 20 Billion parameter models 
using DeepSpeed ZeRO Stage 3 Offload on a single GPU. This does come with a significant throughput hit, 
which needs to be weighed accordingly.
'''
'''
Sharding techniques help when model sizes are fairly large; 
roughly 500M+ parameters is where we’ve seen benefits. 
However, in the following cases, we recommend sticking to ordinary distributed strategies

When your model is small (ResNet50 of around 80M Parameters), 
unless you are using unusually large batch sizes or inputs.

Due to high distributed communication between devices, if running on a slow network/interconnect, 
the training might be much slower than expected and then it’s up to you to determince the tradeoff here.
'''

'''
Cutting-edge and third-party Strategies
Cutting-edge Lightning strategies are being developed by third-parties outside of Lightning.

If you want to try some of the latest and greatest features for model-parallel training, 
check out the Colossal-AI Strategy integration.

Another integration is Bagua Strategy, deep learning training acceleration framework for PyTorch, 
with advanced distributed training algorithms and system optimizations.

For training on unreliable mixed GPUs across the internet check out the Hivemind Strategy integration.
'''
'''
Colossal-AI
https://lightning.ai/docs/pytorch/stable/advanced/third_party/colossalai.html
'''


'''
Fully Sharded Training
https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/#auto-wrapping
'''
'''
Auto Wrapping

While initializing the optimizers inside configure_optimizers hook, make sure to use self.trainer.model.parameters(), else PyTorch will raise an error. This is required because when you use auto-wrap, the model layers are sharded and your lightning_module.parameters() will return a generator with no params. This inconvenience will be addressed in the future.
'''
model = BoringModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="fsdp", precision=16)
trainer.fit(model)

from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   default_auto_wrap_policy,
)
import torch.nn as nn
 
class model(nn.Module):
   def __init__(self):
       super().__init__()
       self.layer1 = nn.Linear(8, 4)
       self.layer2 = nn.Linear(4, 16)
       self.layer3 = nn.Linear(16, 4)
 
model = DistributedDataParallel(model())
fsdp_model = FullyShardedDataParallel(
   model(),
   fsdp_auto_wrap_policy=default_auto_wrap_policy,
   cpu_offload=CPUOffload(offload_params=True),
)


'''
Manual Wrapping
'''
'''
Manual wrapping can be useful to explore complex sharding strategies by applying wrap selectively to some parts of the model. To activate parameter sharding with manual wrapping, you can wrap your model using the wrap function. Internally in Lightning, we enable a context manager around the configure_sharded_model function to make sure the wrap parameters are passed correctly.

When not using Fully Sharded these wrap functions are a no-op. This means once the changes have been made, there is no need to remove the changes for other strategies.

wrap simply wraps the module with a Fully Sharded Parallel class with the correct parameters from the Lightning context manager.

Here’s an example using that uses wrap to create your model:
'''

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torch.distributed.fsdp.wrap import wrap


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(32, 32)
        self.block = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with `wrap`.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        linear_layer = wrap(self.linear_layer)

        for i, layer in enumerate(self.block):
            self.block[i] = wrap(layer)

        self.model = nn.Sequential(linear_layer, nn.ReLU(), self.block)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

from lightning.pytorch.strategies import FSDPStrategy
model = MyModel()
fsdp = FSDPStrategy(cpu_offload=True)
trainer = Trainer(accelerator="gpu", devices=4, strategy=fsdp, precision=16)
trainer.fit(model)

'''
Activation Checkpointing
Activation checkpointing reduces GPU memory usage by avoiding the storage of intermediate activation tensors in selected layers. The tradeoff is that computation cost for the backpropagation increases, as the dropped activations need to be recomputed.

Enable checkpointing on large layers (like Transformers) by providing the layer class/type to the strategy:
'''
from lightning.pytorch.strategies import FSDPStrategy

fsdp = FSDPStrategy(
    activation_checkpointing=MyTransformerBlock,  # or pass a list with multiple types
)
trainer = pl.Trainer(strategy=fsdp, accelerator="gpu", devices=4)



'''
DeepSpeed

'''
'''
DeepSpeed is a deep learning training optimization library, 
providing the means to train massive billion parameter models at scale. 
Using the DeepSpeed strategy, we were able to train model sizes of 10 Billion parameters and above, 
with a lot of useful information in this benchmark and the DeepSpeed docs. 
DeepSpeed also offers lower level training optimizations, 
and efficient optimizers such as 1-bit Adam. 
We recommend using DeepSpeed in environments where speed and memory optimizations are important 
(such as training large billion parameter models).
'''
'''
https://www.deepspeed.ai/tutorials/megatron/
'''
'''
DeepSpeed ZeRO Stage 1 - Shard optimizer states, remains at speed parity with DDP whilst providing memory improvement

DeepSpeed ZeRO Stage 2 - Shard optimizer states and gradients, remains at speed parity with DDP whilst providing even more memory improvement

DeepSpeed ZeRO Stage 2 Offload - Offload optimizer states and gradients to CPU. Increases distributed communication volume and GPU-CPU device transfer, but provides significant memory improvement

DeepSpeed ZeRO Stage 3 - Shard optimizer states, gradients, parameters and optionally activations. Increases distributed communication volume, but provides even more memory improvement

DeepSpeed ZeRO Stage 3 Offload - Offload optimizer states, gradients, parameters and optionally activations to CPU. Increases distributed communication volume and GPU-CPU device transfer, but even more significant memory improvement.

DeepSpeed Activation Checkpointing - Free activations after forward pass. Increases computation, but provides memory improvement for all stages.
'''

'''
DeepSpeed currently only supports single optimizer, single scheduler within the training loop.

When saving a checkpoint we rely on DeepSpeed which saves a directory containing the model and various components.
'''

'''
DeepSpeed ZeRO Stage 1

It is recommended to skip Stage 1 and use Stage 2!!
'''

'''
DeepSpeed ZeRO Stage 2
'''
from lightning.pytorch import Trainer

model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2", precision=16)
trainer.fit(model)

'''
Below we show an example of running ZeRO-Offload. 
ZeRO-Offload leverages the host CPU to offload optimizer memory/computation, 
reducing the overall memory consumption.
'''
from lightning.pytorch import Trainer

model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2_offload", precision=16)
trainer.fit(model)

from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

'''
We suggest tuning the allgather_bucket_size parameter and reduce_bucket_size parameter to find optimum parameters based on your model size. These control how large a buffer we limit the model to using when reducing gradients/gathering updated parameters. Smaller values will result in less memory, but tradeoff with speed.

DeepSpeed allocates a reduce buffer size multiplied by 1.5x so take that into consideration when tweaking the parameters.

The strategy sets a reasonable default of 2e8, which should work for most low VRAM GPUs (less than 7GB), allocating roughly 3.6GB of VRAM as buffer. Higher VRAM GPUs should aim for values around 5e8.
'''
model = MyModel()
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy=DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
    precision=16,
)
trainer.fit(model)


'''
For even more speed benefit, DeepSpeed offers an optimized CPU version of ADAM called DeepSpeedCPUAdam 
to run the offloaded computation, which is faster than the standard PyTorch implementation.
'''
import lightning.pytorch
from lightning.pytorch import Trainer
from deepspeed.ops.adam import DeepSpeedCPUAdam


class MyModel(pl.LightningModule):
    ...

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        return DeepSpeedCPUAdam(self.parameters())


model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2_offload", precision=16)
trainer.fit(model)

'''
DeepSpeed ZeRO Stage 3
'''
'''
DeepSpeed ZeRO Stage 3 shards the optimizer states, gradients and the model parameters (also optionally activations). 
Sharding model parameters and activations comes with an increase in distributed communication, 
however allows you to scale your models massively from one GPU to multiple GPUs. 
The DeepSpeed team report the ability to fine-tune models with over 40B parameters on a single GPU 
and over 2 Trillion parameters on 512 GPUs. 
For more information we suggest checking the DeepSpeed ZeRO-3 Offload documentation.
'''
'''
https://www.deepspeed.ai/2021/03/07/zero3-offload.html
'''
'''
https://github.com/SeanNaren/minGPT/tree/stage3
'''
'''
To reach the highest memory efficiency or model size, you must:

Use the DeepSpeed strategy with the stage 3 parameter

Use CPU Offloading to offload weights to CPU, plus have a reasonable amount of CPU RAM to offload onto

Use DeepSpeed Activation Checkpointing to shard activations

Below we describe how to enable all of these to see benefit. 
With all these improvements we reached 45 Billion parameters training a GPT model on 8 GPUs with ~1TB of CPU RAM available.
'''
'''
Also please have a look at our DeepSpeed ZeRO Stage 3 Tips 
which contains a lot of helpful information when configuring your own models.
'''
'''
https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepspeed-zero-stage-3-tips
'''
from lightning.pytorch import Trainer
from deepspeed.ops.adam import FusedAdam


class MyModel(pl.LightningModule):
    ...

    def configure_optimizers(self):
        return FusedAdam(self.parameters())


model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
trainer.fit(model)

trainer.test()
trainer.predict()

'''
Shard Model Instantly to Reduce Initialization Time/Memory
When instantiating really large models, it is sometimes necessary to shard the model layers instantly.

This is the case if layers may not fit on one single machines CPU or GPU memory, 
but would fit once sharded across multiple machines. 
We expose a hook that layers initialized within the hook will be sharded instantly on a per layer basis,
 allowing you to instantly shard models.

This reduces the time taken to initialize very large models, as well as ensure we do not run out of memory when instantiating larger models. 
For more information you can refer to the DeepSpeed docs for Constructing Massive Models.
'''
'''
https://deepspeed.readthedocs.io/en/latest/zero3.html
'''
import torch.nn as nn
from lightning.pytorch import Trainer
from deepspeed.ops.adam import FusedAdam


class MyModel(pl.LightningModule):
    ...

    def configure_sharded_model(self):
        # Created within sharded model context, modules are instantly sharded across processes
        # as soon as they are made.
        self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

    def configure_optimizers(self):
        return FusedAdam(self.parameters())


model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
trainer.fit(model)

trainer.test()
trainer.predict()


'''
DeepSpeed ZeRO Stage 3 Offload
'''
'''
DeepSpeed ZeRO Stage 3 Offloads optimizer state, 
gradients to the host CPU to reduce memory usage as ZeRO Stage 2 does, 
however additionally allows you to offload the parameters as well for even more memory saving.
'''
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

# Enable CPU Offloading
model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)
trainer.fit(model)

# Enable CPU Offloading, and offload parameters to CPU
model = MyModel()
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy=DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
    ),
    precision=16,
)
trainer.fit(model)


'''
DeepSpeed Infinity (NVMe Offloading)
'''
'''
Additionally, DeepSpeed supports offloading to NVMe drives for even larger models, 
utilizing the large memory space found in NVMes. 
DeepSpeed reports the ability to fine-tune 1 Trillion+ parameters using NVMe Offloading on one 8 GPU machine. 
Below shows how to enable this, 
assuming the NVMe drive is mounted in a directory called /local_nvme
'''
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

# Enable CPU Offloading
model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)
trainer.fit(model)

# Enable CPU Offloading, and offload parameters to CPU
model = MyModel()
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy=DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        remote_device="nvme",
        offload_params_device="nvme",
        offload_optimizer_device="nvme",
        nvme_path="/local_nvme",
    ),
    precision=16,
)
trainer.fit(model)
'''
When offloading to NVMe you may notice that the speed is slow. 
There are parameters that need to be tuned based on the drives that you are using. 
Running the aio_bench_perf_sweep.py script can help you to find optimum parameters. 
See the issue for more information on how to parse the information.
'''


'''
DeepSpeed Activation Checkpointing

'''
'''
Activation checkpointing frees activations from memory as soon as they are not needed during the forward pass. They are then re-computed for the backwards pass as needed.

Activation checkpointing is very useful when you have intermediate layers that produce large activations.

This saves memory when training larger models, however requires using a checkpoint function to run modules as shown below.
'''
from lightning.pytorch import Trainer
import deepspeed


class MyModel(LightningModule):
    ...

    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.block_2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        # Use the DeepSpeed checkpointing function instead of calling the module directly
        # checkpointing self.block_1 means the activations are deleted after use,
        # and re-calculated during the backward passes
        x = deepspeed.checkpointing.checkpoint(self.block_1, x)
        return self.block_2(x)


'''

DeepSpeed ZeRO Stage 3 Tips !!!!!!!!!!!!!!!




Here is some helpful information when setting up DeepSpeed ZeRO Stage 3 with Lightning.

If you’re using Adam or AdamW, ensure to use FusedAdam or DeepSpeedCPUAdam (for CPU Offloading) rather than the default torch optimizers as they come with large speed benefits

Treat your GPU/CPU memory as one large pool. In some cases, you may not want to offload certain things (like activations) to provide even more space to offload model parameters

When offloading to the CPU, make sure to bump up the batch size as GPU memory will be freed

We also support sharded checkpointing. By passing save_full_weights=False to the DeepSpeedStrategy, we’ll save shards of the model which allows you to save extremely large models. 
However to load the model and run test/validation/predict you must use the Trainer object.
'''

'''
Collating Single File Checkpoint for DeepSpeed ZeRO Stage 3
'''
'''
After training using ZeRO Stage 3, you’ll notice that your checkpoints are a directory of sharded model and optimizer states. 
If you’d like to collate a single file from the checkpoint directory please use the below command, 
which handles all the Lightning states additionally when collating the file.
'''
'''
This single file checkpoint does not include the optimizer/lr-scheduler states. 
This means we cannot restore training via the trainer.fit(ckpt_path=) call. 
Ensure to keep the sharded checkpoint directory if this is required.
'''
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# lightning deepspeed has saved a directory instead of a file
save_path = "lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt/"
output_path = "lightning_model.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

'''
Custom DeepSpeed Config
'''
'''
In some cases you may want to define your own DeepSpeed Config, to access all parameters defined. 
We’ve exposed most of the important parameters, however, there may be debugging parameters to enable. 
Also, DeepSpeed allows the use of custom DeepSpeed optimizers and schedulers defined within a config file that is supported.
'''
'''
All strategy default parameters will be ignored when a config object is passed. 
All compatible arguments can be seen in the DeepSpeed docs.
'''
'''
https://www.deepspeed.ai/docs/config-json/
'''
deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 3e-5,
            "betas": [0.998, 0.999],
            "eps": 1e-5,
            "weight_decay": 1e-9,
            "cuda_aware": True,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "last_batch_iteration": -1,
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 100,
        },
    },
    "zero_optimization": {
        "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        "offload_optimizer": True,  # Enable Offloading optimizer state/calculation to the host CPU
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
        "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
    },
}

model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy=DeepSpeedStrategy(config=deepspeed_config), precision=16)
trainer.fit(model)

from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

model = MyModel()
trainer = Trainer(
    accelerator="gpu", devices=4, strategy=DeepSpeedStrategy(config="/path/to/deepspeed_config.json"), precision=16
)
trainer.fit(model)