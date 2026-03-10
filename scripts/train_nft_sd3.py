# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters, # theta the trainable 'default' lora adapter group
):
    import ipdb; ipdb.set_trace()
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True) # NOTE  copy ema's cached parameters to current 'transformer_trainable_parameters' TODO 1. transformer_trainable_parameters -> ema's temp cache = ema's temp_stored_parameters; 2. ema's stored ema.ema_parameters --> copy to -> transformer_trainable_parameters/theta
    ### 相当于是在说，这里的evaluate，实际上用的是来自ema里面的cache的参数集合了，也就是：ema.ema_parameters!!! NOTE NOTE NOTE
    pipeline.transformer.eval() # TODO 如何验证这里的pipeline里面使用的参数，是来自ema.ema_parameters-> theta的那个theta呢？？？

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

        break # NOTE TODO for debug only

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    import ipdb; ipdb.set_trace()
    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters) # ema's temp_stored_parameters -> theta, recover pipeline

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        import ipdb; ipdb.set_trace()
        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True) # NOTE 所以说，实际save到hard disk的是ema.ema_parameters，不是当前的theta=transformer_trainable_parameters

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        import ipdb; ipdb.set_trace()
        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters) # ema's temp_stored_parameters -> theta
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config
    #import ipdb; ipdb.set_trace()
    # --- Distributed Setup ---
    #rank = int(os.environ["RANK"])
    #world_size = int(os.environ["WORLD_SIZE"])
    #local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="flow-grpo", name=config.run_name, config=config.to_dict(), dir=log_dir)
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)
    #import ipdb; ipdb.set_trace()
    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        #config.pretrained.model
        '/workspace/asr/flow_grpo/ckpts/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80',
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora: # True
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path: # None
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config) # NOTE here, add 'old' and 'default' adapter types, each adapter type is with 18,776,064=18.7M parameters.
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer, 'default' lora adapter group
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval": # NOTE here
        train_dataset = GenevalPromptDataset(config.dataset, "train") # /workspace/asr/DiffusionNFT/dataset/geneval; with 50k samples
        test_dataset = GenevalPromptDataset(config.dataset, "test") # with 2211 samples
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size, 9
        k=9, #config.sample.num_image_per_prompt, # 24 TODO
        num_replicas=world_size, # 1
        rank=rank, # 0
        seed=config.seed, # 42
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    #import ipdb; ipdb.set_trace()
    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation
    #import ipdb; ipdb.set_trace()
    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch # 9 * 1 * 16 = 144
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            transformer_ddp.module.load_adapter(lora_path, adapter_name="old", is_trainable=False)
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(f"Resumed global_step to {global_step}. Epoch estimation might be needed.")
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    import ipdb; ipdb.set_trace()
    ema = None
    if config.train.ema: # True
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data) # src_param --> tgt_param i.e., theta --> theta^old, update theta^old by theta NOTE
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs): # first_epoch=0, config.num_epochs=100000
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch), # 16
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            #import ipdb; ipdb.set_trace()
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter) # len(prompts)=9, 'prompt' in 'prompts' is all the same, 重复了9次而已, e.g., prompts[0]='A vibrant digital banner on a website header, prominently displaying the text "Limited Time Offer" in bold, eye-catching colors, set against a dynamic background with subtle gradients and modern design elements.'

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline, # <class 'diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline'> 'default' lora adapter NOTE
                    test_dataloader, # 这是每次搞了policy update之后，都重新评估一下pipeline这个模型的质量
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn, # <function multi_score.<locals>._fn at 0x7fe4093b2a70>
                    executor, # <concurrent.futures.thread.ThreadPoolExecutor object at 0x7fe9ebe60580>
                    mixed_precision_dtype, # torch.float16
                    ema, # <flow_grpo.ema.EMAModuleWrapper object at 0x7fe409225c30>
                    transformer_trainable_parameters, # 18,776,064=18.7M 可训练参数; a list with 382 tensors
                ) # NOTE 这里只是调用了eval_fn一次，没有其他的地方的调用了

            if i == 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer_ddp, # both 'old' lora adapter and 'default' lora adapter are saved to hard disk
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            #import ipdb; ipdb.set_trace()
            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline, # <class 'diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline'> NOTE 'old' lora adapter
                        prompt_embeds=prompt_embeds, # [9, 205, 4096]
                        pooled_prompt_embeds=pooled_prompt_embeds, # [9, 2048]
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)], # [9, 205, 4096], prune by len(prompts) -> no change
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)], # [9, 2048], prune by len(prompts) -> no change
                        num_inference_steps=config.sample.num_steps, # 10
                        guidance_scale=config.sample.guidance_scale, # 1.0
                        output_type="pt",
                        height=config.resolution, # 512
                        width=config.resolution, # 512
                        noise_level=config.sample.noise_level, # 0.7
                        deterministic=config.sample.deterministic, # True
                        solver=config.sample.solver, # 'dpm2'
                        model_type="sd3",
                    ) # latents=a list with 11 tensors, latents[0].shape=[9, 16, 64, 64] NOTE 这是使用任意solver，来通过多次迭代，从x1 -> x0 =images的方法。这个方法负责的是：prompt -> flow-matching model + solver with 10 iterations -> image; images.shape=[9, 3, 512, 512], 
            transformer_ddp.module.set_adapter("default")
            #import ipdb; ipdb.set_trace()

            latents = torch.stack(latents, dim=1) # after stack, latents.shape=[9, 11, 16, 64, 64]
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0) # 让当前线程“主动让出 CPU”，把执行权交给操作系统调度器，然后立刻重新进入可调度状态。
            # submit任务 -> 让出cpu -> worker线程启动reward_fn -> 主线程继续
            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids, # [9, 256]
                    "prompt_embeds": prompt_embeds, # [9, 205, 4096]
                    "pooled_prompt_embeds": pooled_prompt_embeds, # [9, 2048]
                    "timesteps": timesteps, # [9, 10], e.g., tensor([[1000.0000,  960.1293,  913.3490,  857.6923,  790.3683,  707.2785,           602.1506,  464.8760,  278.0488,    8.9286],
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1), # [9, 9] + [9, 1] -> [9, 10], e.g., tensor([[960.1293, 913.3490, 857.6923, 790.3683, 707.2785, 602.1506, 464.8760,          278.0488,   8.9286,   0.0000],
                    "latents_clean": latents[:, -1], # NOTE textual prompt -> pi^old -> x0=latents[:, -1] 取的是trajectory中的最后一个，也就是我们想要的结果
                    "rewards_future": rewards_future,  # Store future
                }
            )
        import ipdb; ipdb.set_trace()
        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )
        #import ipdb; ipdb.set_trace()
        if config.per_prompt_stat_tracking: # True
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"]) # [144, 9] # NOTE this is using grpo algorithm to compute advantages=(r-mean)/std

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps # 8*9=72

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs): # 1
            import ipdb; ipdb.set_trace()
            perm = torch.randperm(total_batch_size_filtered, device=device) # 144, 生成0-143之间的一个全排列
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()} # dict_keys(['prompt_embeds', 'pooled_prompt_embeds', 'timesteps', 'next_timesteps', 'latents_clean', 'advantages'])=filtered_samples.keys()

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            ) # [144, 10], e.g., tensor([[0, 2, 9,  ..., 7, 4, 3], -> tensor([0, 2, 9, 8, 1, 6, 5, 7, 4, 3], device='cuda:0')
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ] # since both 'timesteps' and 'next_timesteps' use the same 'perms_time', so the relative order keeps the same, that is shuffled_filtered_samples['timesteps'] > shuffled_filtered_samples['next_timesteps']

            training_batch_size = total_batch_size_filtered // num_batches # 144//16=9

            samples_batched_list = []
            for k_batch in range(num_batches): # 16
                batch_dict = {}
                start = k_batch * training_batch_size # 0
                end = (k_batch + 1) * training_batch_size # 9
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)
            # NOTE 上面这个是把[144, 205, 4096], 按照batch-size=16，来切分成9个batches
            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)), # NOTE with 16 batches
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                #import ipdb; ipdb.set_trace()
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"]) # 9

                if config.sample.guidance_scale > 1.0: # = 1.0 NOTE meaningful only when >1.0, noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond), when self.guidance_scale=1.0, then noise_pred = noise_pred_uncond + noise_pred_text - noise_pred_uncond = noise_pred_text; 此外，如果guidance_scale < 1.0，则cfg占据了上风，没有意义了.
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"] # [9, 205, 4096]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"] # [9, 2048]

                # Loop over timesteps (NOTE all T timesteps) for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)), # 9
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    #import ipdb; ipdb.set_trace()
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"] ### 1 x0: [9, 16, 64, 64], prompt -> pi^old -> trajectory[-1] = predicted x0 image related tensor

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0 # NOTE /1000 again to a range of [0, 1] now; before in calling transformer (FM), used 1000 * t; t=tensor([1.0000, 0.2780, 0.7073, 0.7073, 0.7073, 0.6022, 0.9133, 0.2780, 0.2780], pick one timestep for one sequence/sample (totally 9 samples)

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1))) ### 2 t: shape from [9] to [9, 1, 1, 1]

                    noise = torch.randn_like(x0.float()) ### 3 epsilon

                    xt = (1 - t_expanded) * x0 + t_expanded * noise ### 4 xt, NOTE t这里有随机了!!! 每个序列的可能不一样了 t_expanded.shape=[9, 1, 1, 1], x0.shape=[9, 16, 64, 64], noise.shape=[9, 16, 64, 64]

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old") # NOTE why? what is 'old' for? -> pi^old in the paper
                        with torch.no_grad():
                            # prediction v^old 无梯度
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx], # j_idx 是batch中一个sample的索引号
                                encoder_hidden_states=embeds, # [9, 205, 4096]
                                pooled_projections=pooled_embeds, # [9, 2048]
                                return_dict=False,
                            )[0].detach() ### 5 v_pred_old, with 'old' adapter
                        transformer_ddp.module.set_adapter("default") 

                        # prediction v 有梯度
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0] ### 6 v_pred_default=v_theta, velocity field vector with lora adapters - the policy on focus for current updating, with 'default' lora adapters? -> trainable!!!

                        with torch.no_grad():  # Reference model part 无梯度
                            # For LoRA, disable adapter.
                            if config.use_lora: # True
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0] ### 7 v_ref, real ref since disabled lora adapters!
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx], # NOTE TODO j_idx is alike timestep, the bad problem is that, advantage is timestep insensitive... e.g., tensor([-0.5167, -0.5167, -0.5167, -0.5167, -0.5167, -0.5167, -0.5167, -0.5167, -0.5167], device='cuda:0', dtype=torch.float64), 9 timesteps
                        -config.train.adv_clip_max, # -5
                        config.train.adv_clip_max, # 5
                    ) ### 8 advantages_clip
                    if hasattr(config.train, "adv_mode"): # NOTE config.train.adv_mode='all'
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5 ### 9 normalized_advantages_clip TODO why? which part in the paper? https://arxiv.org/pdf/2509.16117 page 6 in r(x_0, c), config.train.adv_clip_max=5
                    r = torch.clamp(normalized_advantages_clip, 0, 1) ### 10 r for reward
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach() ### 11
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach() ### 12
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach() ### 13 deviate = pian li = away from
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach() ### 14
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach() ### 15 NOTE v^+_theta right below Equation 5 in page 5 of the paper, 'implicit positive policy' line-10 in algorithm 1; TODO 因为old_prediction是torch.no_grad()下计算出来的，所以其没有被加入计算图，所以也不需要detach()! 待定
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction ### 16 NOTE v^-_theta 'implicit negative policy' in the paper, line-11 in algorithm 1

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction ### 17 NOTE page 6's x_theta = x_t - t * v_theta, where x_theta = x0_prediction, 预测出来的x0 = xt带噪图片 - t*速度场向量，这个v^_theta是noise - data，即从data指向noise的！所以，这里noise - v^_theta = data = x0_prediction得到的是从xt一步“去噪”之后，得到的预测出来的real data x0
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        ) ### 18 weight_factor.shape=[9, 1, 1, 1] each sample with one weight factor
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim))) ### 19, shape=[9]
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction ### 20 NOTE alike line-953
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        ) ### 21
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    ) ### 22 NOTE same format alike line 960

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta ### 23 equation 5 ? why added 'config.beta=0.1'??? shape=[9]
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean() ### 24 config.train.adv_clip_max=5

                    loss = policy_loss ### 25
                    loss_terms["policy_loss"] = policy_loss.detach() ### 26
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach() ### 27

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )### 28 ref_forward_prediction is the mean? kl_div_loss.shape=[9]

                    loss += config.train.beta * torch.mean(kl_div_loss) ### 29 config.train.beta=0.0001
                    kl_div_loss = torch.mean(kl_div_loss) # mean([9]) --> scalar
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach() ### 30
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach() ### 31
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach() ### 32

                    loss_terms["total_loss"] = loss.detach() ### 33
                    #import ipdb; ipdb.set_trace()
                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps ### 34 effective_grad_accum_steps=72
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation # NOTE
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle
                import ipdb; ipdb.set_trace()
                if (
                    config.train.ema # True
                    and ema is not None # True
                    and (current_accumulated_steps % effective_grad_accum_steps == 0) # 9 % 72 = 9
                ):
                    import ipdb; ipdb.set_trace() # 8 batches * 9 timesteps = 72; NOTE totally 16 batches, so, two ema updates 
                    ema.step(transformer_trainable_parameters, global_step) # global_step=1

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
