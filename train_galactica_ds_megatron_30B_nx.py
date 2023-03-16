from transformers import (
    TrainingArguments,
    AutoTokenizer,
    OPTForCausalLM
    )
import torch
import argparse
import time
from build_index_mappings import _build_train_valid_test_datasets
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

parser = argparse.ArgumentParser()
parser.add_argument("--student_model_path", default="/data/FlagAI/examples/gpt3_pretrain/state_dict/galactica-30b-en", type=str, help="path of the pretrained student model")
parser.add_argument("--save_path", default="outputs/run_galactica", type=str, help="path to save checkpoints")
parser.add_argument("--train_file", default="/data/dedup_wudao/dedup_wudao_5pct_merged_text_document", type=str, help="path to train file")
parser.add_argument("--host_file", default="hostfiles", type=str, help="path to eval file")
parser.add_argument("--data_scripts", default="data_script/json.py", type=str, help="path to save checkpoints")
parser.add_argument("--data_cache", default="data_cache/", type=str, help="path to save checkpoints")
parser.add_argument("--log_dir", default="logs/", type=str, help="path to save checkpoints")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=1, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=1.0e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The episilon for Adam.")
parser.add_argument("--clip_grad_norm", default=1.0, type=float, help="The truncated grad threshold")
parser.add_argument("--log_every_n_step", default=64, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_steps", default=6400, type=int, help="The epochs to save the trained models.")
parser.add_argument("--max_seq_length", default=2048, type=int, help="The maximum length of input sentences")
parser.add_argument("--gradient_accumulation_steps", default=64, type=int, help="The gradient_accumulation_steps.")
parser.add_argument("--not_call_launch", default=False, action="store_true")
parser.add_argument("--world_size", default=16, type=int)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()

def main():

    trainer = Trainer(
        env_type="deepspeed+mpu",
        epochs=args.epochs,
        batch_size=args.per_gpu_batch_size,
        eval_interval=1e12,
        log_interval=args.gradient_accumulation_steps,
        save_interval=args.save_every_n_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        experiment_name='galactica-30B',
        load_dir=None,
        lr=args.learning_rate,
        master_ip='127.0.0.1',
        num_gpus=8,
        num_nodes=1,
        master_port=29513,
        fp16=True,
        checkpoint_activations=True,
        hostfile=args.host_file,
        training_script=__file__,
        deepspeed_config='deepspeed_30B.json',
        model_parallel_size = 8,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
    model_dir = "/data/FlagAI/examples/gpt3_pretrain/state_dict"
    auto_loader = AutoLoader(
    "lm",
    model_name="galactica-30b-en",
    model_dir=model_dir,
    )
    model = auto_loader.get_model()

    # TODO
    ### 需要根据数据集情况填写路径前缀
    data_prefix = args.train_file
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [10500000, 1000, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    train_dataset, eval_dataset, test_dataset = _build_train_valid_test_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup)

    def distill_collate_fn(batch):
        def padding(indice, max_length, pad_idx=1):
            pad_indice = [
                item.tolist() + [1] * max(0, max_length - len(item.tolist())) for item in indice
            ]
            return torch.tensor(pad_indice)

        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length)[:,:seq_length]

        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data

    trainer.train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=eval_dataset,
        collate_fn=distill_collate_fn,
    )

if __name__ == '__main__':
    main()
