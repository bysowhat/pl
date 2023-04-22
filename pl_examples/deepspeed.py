'''
DeepSpeed: Getting Started Page https://www.deepspeed.ai/getting-started/

ZeRO-3 Offload Documentation, Tutorial
https://deepspeed.readthedocs.io/en/latest/zero3.html
https://www.deepspeed.ai/tutorials/ZeRO/#training-trillion-scale-models-with-ZeRO-3-offload
'''


'''
deepspeed.initialize ensures that all of the necessary setup required for distributed data parallel 
or mixed precision training are done appropriately under the hood. In addition to wrapping the model, 
DeepSpeed can construct and manage the training optimizer, data loader, 
and the learning rate scheduler based on the parameters passed to deepspeed.initialize and the DeepSpeed configuration file. 
Note that DeepSpeed automatically executes the learning rate schedule at every training step
'''
#To initialize the DeepSpeed engine:
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)

'''
Under the hood, DeepSpeed automatically performs the necessary operations required for distributed data parallel training, in mixed precision, with a pre-defined learning rate scheduler:

Gradient Averaging: in distributed data parallel training, backward ensures that gradients are averaged across data parallel processes after training on an train_batch_size.

Loss Scaling: in FP16/mixed precision training, the DeepSpeed engine automatically handles scaling the loss to avoid precision loss in the gradients.

Learning Rate Scheduler: when using a DeepSpeed’s learning rate scheduler (specified in the ds_config.json file), DeepSpeed calls the step() method of the scheduler at every training step (when model_engine.step() is executed). When not using DeepSpeed’s learning rate scheduler:

if the schedule is supposed to execute at every training step, then the user can pass the scheduler to deepspeed.initialize when initializing the DeepSpeed engine and let DeepSpeed manage it for update or save/restore.
if the schedule is supposed to execute at any other interval (e.g., training epochs), then the user should NOT pass the scheduler to DeepSpeed during initialization and must manage it explicitly.
'''
#Training
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()


'''
Saving and loading the training state is handled via the save_checkpoint and load_checkpoint API in DeepSpeed which takes two arguments to uniquely identify a checkpoint:

ckpt_dir: the directory where checkpoints will be saved.
ckpt_id: an identifier that uniquely identifies a checkpoint in the directory. In the following code snippet, we use the loss value as the checkpoint identifier.
'''
'''
DeepSpeed can automatically save and restore the model, optimizer, and the learning rate scheduler states while hiding away these details from the user. However, the user may want to save additional data that are unique to a given model training. To support these items, save_checkpoint accepts a client state dictionary client_sd for saving. These items can be retrieved from load_checkpoint as a return argument. In the example above, the step value is stored as part of the client_sd.

Important: all processes must call this method and not just the process with rank 0. It is because each process needs to save its master weights and scheduler+optimizer states. This method will hang waiting to synchronize with other processes if it’s called just for the process with rank 0.
'''
#Model Checkpointing
#load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

#advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):

    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

    #save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)


'''
DeepSpeed features can be enabled, disabled, or configured using a config JSON file 
that should be specified as args.deepspeed_config. 
A sample config file is shown below. For a full set of features see API doc.
'''
#DeepSpeed Configuration
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}


'''
ZeRO-Infinity and ZeRO-Offload work best with our heavily optimized deepspeed.ops.adam.DeepSpeedCPUAdam optimizer. 
We recommend using our optimizer config to instruct deepspeed.initialize() to build the optimizer for you.
'''


#Example ZeRO-3 Configurations
#Use ZeRO to partition the optimizer states (stage 1), gradients (stage 2), and parameters (stage 3).

{
    "zero_optimization": {
        "stage": 3,
    },
    "fp16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
        "lr": 0.001,
        "betas": [
            0.8,
            0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },
    ...
}
#Additionally offload the optimizer states and computations to the CPU with ZeRO-Infinity.

{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    ...
}
#Save even more memory by offloading parameters to the CPU memory.

{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        }
        "offload_param": {
            "device": "cpu"
        }
    },
    ...
}
#Save even MORE memory by offloading to NVMe (if available on your system):

{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/nvme_data"
        }
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/nvme_data"
        }
    },
    ...
}


#Assumptions
'''
The forward and backward passes of submodules must individually fit in device memory. 
If this not the case, deepspeed.zero.
TiledLinear implements memory-centric tiling and works with ZeRO-3 to break linear layers into a sequence of smaller submodules 
that can fit in memory.

A module’s parameters are only accessed within its own __init__ and forward() methods. Otherwise, 
DeepSpeed must be instructed to collect and re-partition the parameter. 
See Manual Parameter Coordination for manually coordinating parameters.
'''


#Constructing Massive Models
#Registering External Parameters
#Gathering Parameters


#Memory-Centric Tiling
'''
To reduce the working memory requirements of DL training for large models, 
ZeRO-Infinity includes technique called memory-centric tiling that exploits the data fetch and release pattern of ZeRO-3 
to reduce the working memory requirements by breaking down a large operator into smaller tiles that can be executed sequentially. 
When combined with ZeRO-3, the parameter and gradients of each tile can be fetched and released one at a time, 
reducing the working memory proportional to the number of tiles. Therefore, ZeRO-Infinity can support operators of arbitrary sizes,
 without refactoring for model parallelism to fit them in limited GPU memory.
'''
