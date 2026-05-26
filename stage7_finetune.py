"""
Stage 7: Fine-tune SPECTER2 on triplets using TripletLoss.
Integrated with platform_sdk for experiment tracking.
"""
import json
import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from platform_sdk import Run

# ── Config ────────────────────────────────────────────────────────────────────
TRIPLETS_FILE = "triplets.jsonl"
OUTPUT_DIR    = "./specter2-finetuned"
BASE_MODEL    = "allenai/specter2_base"
EPOCHS        = 3
BATCH_SIZE    = 16
WARMUP_STEPS  = 500
MAX_SEQ_LEN   = 256
TRAIN_LIMIT   = 200_000

cfg = {
    "base_model":   BASE_MODEL,
    "epochs":       EPOCHS,
    "batch_size":   BATCH_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "max_seq_len":  MAX_SEQ_LEN,
    "train_limit":  TRAIN_LIMIT,
    "loss":         "TripletLoss",
    "mixed_prec":   "FP16",
}
# ─────────────────────────────────────────────────────────────────────────────

with Run(name="cadence-specter2-finetune", config=cfg, tags=["cadence", "specter2", "nlp"]) as run:

    run.enable_gpu_profiler(threshold_pct=80)

    # ── GPU check ─────────────────────────────────────────────────────────────
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # ── Load triplets ─────────────────────────────────────────────────────────
    print(f"\nLoading triplets from {TRIPLETS_FILE}...")
    examples = []

    with open(TRIPLETS_FILE) as f:
        for line in tqdm(f, desc="Loading"):
            t = json.loads(line)
            examples.append(InputExample(texts=[
                t["anchor"][:MAX_SEQ_LEN * 4],
                t["positive"][:MAX_SEQ_LEN * 4],
                t["negative"][:MAX_SEQ_LEN * 4],
            ]))
            if len(examples) >= TRAIN_LIMIT:
                break

    print(f"Loaded {len(examples):,} triplets")
    run.log({"triplets_loaded": len(examples)}, step=0)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model: {BASE_MODEL}...")
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LEN

    # ── Step counter (mutable list so closure can modify it) ──────────────────
    step_counter = [0]

    def step_callback(score, epoch, steps):
        run.log({"loss": score, "epoch": epoch}, step=step_counter[0])
        run.profiler.check(step_counter[0])
        step_counter[0] += 1

    # ── Training ──────────────────────────────────────────────────────────────
    loader          = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fn         = losses.TripletLoss(model=model)
    steps_per_epoch = len(loader)

    print(f"\nStarting fine-tuning...")
    print(f"  Triplets     : {len(examples):,}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Steps/epoch  : {steps_per_epoch:,}\n")

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=OUTPUT_DIR,
        show_progress_bar=True,
        use_amp=True,
        callback=step_callback,
        checkpoint_path=OUTPUT_DIR,
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=2,
    )

    # ── Log epoch metrics ─────────────────────────────────────────────────────
    run.log_metrics({
        "epochs_completed": EPOCHS,
        "total_steps":      step_counter[0],
        "triplets_trained": len(examples),
    })

    # ── Save model artifact ───────────────────────────────────────────────────
    print(f"\nSaving model artifact...")
    run.log_artifact(OUTPUT_DIR, kind="checkpoint")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nRunning sanity check...")
    test_papers = [
        "Attention is all you need. We propose a transformer architecture based on attention.",
        "BERT: Pre-training deep bidirectional transformers for language understanding.",
        "A study of fatty acid metabolism in breast cancer cells via ACSL4 expression.",
    ]

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = model.encode(test_papers, device=device)
    sim        = cosine_similarity(embeddings)

    attn_bert   = float(sim[0][1])
    attn_cancer = float(sim[0][2])
    gap         = attn_bert - attn_cancer

    print(f"  Attention ↔ BERT   : {attn_bert:.3f}  (should be HIGH)")
    print(f"  Attention ↔ Cancer : {attn_cancer:.3f}  (should be LOW)")
    print(f"  Gap                : {gap:.3f}  (should be > 0.05)")

    run.log_metrics({
        "sanity_attn_bert":   attn_bert,
        "sanity_attn_cancer": attn_cancer,
        "sanity_gap":         gap,
    })

    if gap > 0.05:
        print("\n✅ Sanity check passed")
    else:
        print("\n⚠️  Sanity check weak — consider more epochs")

print(f"\nDone. Model saved to {OUTPUT_DIR}/")