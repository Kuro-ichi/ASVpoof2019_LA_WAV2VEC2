from __future__ import annotations
import argparse, os, torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AdamW, get_linear_schedule_with_warmup
from wav2vec2_xlsr.config import add_common_args, load_yaml, merge_overrides
from wav2vec2_xlsr.data.dataset import ASRDataset
from wav2vec2_xlsr.models.asr import build_asr
from wav2vec2_xlsr.utils.logger import SimpleLogger
from wav2vec2_xlsr.utils.seed import set_seed

def collate_fn(batch, processor):
    # batch: list of dicts with 'input_values' (Tensor[T]) and 'text' (str)
    inputs = [b["input_values"] for b in batch]
    texts  = [b["text"] for b in batch]
    # processor expects list of arrays; pad to longest
    inputs = [x.numpy() for x in inputs]
    batch_inputs = processor(inputs, sampling_rate=processor.feature_extractor.sampling_rate,
                             return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(texts, return_tensors="pt", padding=True).input_ids
    batch_inputs["labels"] = labels
    return batch_inputs, texts

def train_loop(model, processor, train_loader, device, epochs, lr, weight_decay, grad_accum=1):
    logger = SimpleLogger()
    model.to(device)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader) // grad_accum
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    model.train()
    step = 0
    for epoch in range(1, epochs+1):
        for (batch, _) in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            if step % 10 == 0:
                logger.log(epoch=epoch, step=step, loss=f"{loss.item():.4f}")
            step += 1

def main(argv=None):
    p = argparse.ArgumentParser("Wav2Vec2 Trainer")
    add_common_args(p)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    cfg = merge_overrides(cfg, args.override)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg.get("model",{}).get("name","facebook/wav2vec2-base-960h")
    sample_rate = cfg.get("data",{}).get("sample_rate",16000)
    bs = cfg.get("train",{}).get("batch_size",8)
    nw = cfg.get("train",{}).get("num_workers",2)
    epochs = cfg.get("train",{}).get("epochs",3)
    lr = cfg.get("train",{}).get("lr",1e-4)
    weight_decay = cfg.get("train",{}).get("weight_decay",0.0)
    grad_accum = cfg.get("train",{}).get("grad_accum",1)
    save_dir = cfg.get("train",{}).get("ckpt_dir","./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Build ASR
    bundle = build_asr(model_name)
    processor, model = bundle.processor, bundle.model

    # Datasets
    if not args.train_csv:
        raise SystemExit("Please provide --train_csv path to CSV manifest")
    train_ds = ASRDataset(args.train_csv, sample_rate=sample_rate)
    collate = lambda b: collate_fn(b, processor)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate)

    # Train
    train_loop(model, processor, train_loader, device, epochs, lr, weight_decay, grad_accum)

    # Save
    path = os.path.join(save_dir, "asr.pth")
    torch.save({"model": model.state_dict(), "processor": processor.to_dict() if hasattr(processor, 'to_dict') else None}, path)
    print(f"Saved checkpoint to {path}")

if __name__ == "__main__":
    main()
