import torch
import os
# PyTorch utils

def train_model(model, dataloader, num_epochs, optimizer, loss_fn, callbacks):
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def setup_device(current_gpu_index, num_gpus):
    # Device setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    device = torch.device("cuda:{}".format(current_gpu_index))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus,
        rank=current_gpu_index,
    )
    torch.cuda.set_device(device)


def cleanup():
    torch.distributed.destroy_process_group()

def prepare_dataloader(dataset, current_gpu_index, num_gpus, batch_size):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        rank=current_gpu_index,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader

def per_device_launch_fn(current_gpu_index, num_gpu):
    # Setup the process groups
    setup_device(current_gpu_index, num_gpu)

    dataset = get_dataset()
    model = get_model()

    # prepare the dataloader
    dataloader = prepare_dataloader(dataset, current_gpu_index, num_gpu, batch_size)

    # Put model on device
    model = model.to(current_gpu_index)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[current_gpu_index], output_device=current_gpu_index
    )

    train_model(ddp_model, dataloader, num_epochs, optimizer, loss_fn)

    cleanup()