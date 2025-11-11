from __future__ import annotations
import argparse, os, torch
from torch.utils.data import DataLoader
from wav2vec2_xlsr.config import add_common_args, load_yaml, merge_overrides
from wav2vec2_xlsr.data.dataset import ASRDataset
from wav2vec2_xlsr.models.asr import build_asr
from wav2vec2_xlsr.utils.metrics import compute_wer

def collate_fn(batch, processor):
    inputs = [b["input_values"].numpy() for b in batch]
    texts  = [b["text"] for b in batch]
    batch_inputs = processor(inputs, sampling_rate=processor.feature_extractor.sampling_rate,
                             return_tensors="pt", padding=True)
    return batch_inputs, texts

def greedy_decode(logits, processor):
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)

def main(argv=None):
    p = argparse.ArgumentParser("Wav2Vec2 Evaluator")
    add_common_args(p)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    cfg = merge_overrides(cfg, args.override)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg.get("model",{}).get("name","facebook/wav2vec2-base-960h")
    sample_rate = cfg.get("data",{}).get("sample_rate",16000)
    if not args.val_csv:
        raise SystemExit("Please provide --val_csv path to CSV manifest")
    if not args.checkpoint:
        raise SystemExit("Please provide --checkpoint path to checkpoint")

    # Build model/processor
    bundle = build_asr(model_name)
    processor, model = bundle.processor, bundle.model
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Data
    ds = ASRDataset(args.val_csv, sample_rate=sample_rate)
    collate = lambda b: collate_fn(b, processor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    refs, hyps = [], []
    with torch.no_grad():
        for batch, texts in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            pred = greedy_decode(out.logits, processor)
            refs.extend(texts)
            hyps.extend(pred)

    w = compute_wer(refs, hyps)
    print(f"WER: {w:.4f} (n={len(refs)})")

if __name__ == "__main__":
    main()
