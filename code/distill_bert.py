# =============================================================================
# distill_bert.py  --  Stages D1 . D2 . D3 . D4 . D5
#
# Creates a mobile-deployable text encoder for surgical procedure descriptions.
# Trains a tiny domain-specific transformer (< 6 MB fp32 / ~1.5 MB INT8) to
# mimic Bio_ClinicalBERT (400 MB) via knowledge distillation.
#
# Stages:
#   D1 -- Load corpus + teacher embeddings (ClinicalBERT > SentenceBERT > fresh)
#   D2 -- Build surgical BPE vocabulary  (vocab_size=4096, corpus-specific)
#   D3 -- Knowledge distillation: train TinySurgicalBERT
#   D4 -- INT8 quantisation + ONNX export  (mobile-ready)
#   D5 -- Generate + validate final embeddings, save pipeline-compatible cache
#
# Prerequisites:
#   pipeline.py Stage 01 complete  (./data/surgical_data.db)
#   pip install torch tokenizers onnx onnxruntime scikit-learn
#
# Outputs:
#   ./models/surgical_tiny_bert/              PyTorch model + BPE tokenizer
#   ./models/surgical_tiny_bert_int8.pt       INT8 quantised PyTorch model
#   ./models/surgical_tiny_bert.onnx          ONNX export (fp32)
#   ./models/surgical_tiny_bert_q8.onnx       INT8 ONNX  (mobile-ready)
#   ./data/bert_cache/tinybert_scheduled_procedure.npy  pipeline-compatible cache
#
# Pipeline Integration (after running this script):
#   In pipeline.py, add to S02_TASKS:
#       3: ('tinybert', 'tinybert_scheduled_procedure.npy')
#   Add 'tinybert' to BERT_ENCODINGS.
#   Change FEATURES_PER_COL to [10, 50, 100, 200] -- tinybert output is 256-d,
#   so all PCA targets are valid.
#
# Mobile Inference:
#   The ONNX model accepts:
#       input_ids      int64 [batch, seq_len]   -- token IDs from BPE tokenizer
#       attention_mask int64 [batch, seq_len]   -- 1 for real tokens, 0 for pad
#   Returns:
#       embeddings     float32 [batch, 256]
# =============================================================================

# =============================================================================
# +==========================================================================+
# |  CONFIG  --  edit only this block                                        |
# +==========================================================================+
# =============================================================================

# -- Paths (must match pipeline.py) --------------------------------------------
DB_PATH    = './data/surgical_data.db'
BERT_DIR   = './data/bert_cache'
MODEL_DIR  = './models/surgical_tiny_bert'
LOG_DIR    = './results'

CLEAN_TABLE = 'Clean'
TEXT_COL    = 'scheduled_procedure'
TARGET      = 'actual_casetime_minutes'

# Teacher cache paths (checked in priority order)
TEACHER_CLINICALBERT  = './data/bert_cache/clinicalbert_scheduled_procedure.npy'
TEACHER_SENTENCEBERT  = './data/bert_cache/sentencebert_scheduled_procedure.npy'
SENTENCEBERT_MODEL_ID = 'all-MiniLM-L6-v2'
CLINICALBERT_MODEL_ID = 'emilyalsentzer/Bio_ClinicalBERT'

# Output paths
OUT_NPY     = './data/bert_cache/tinybert_scheduled_procedure.npy'
OUT_PT_FP32 = './models/surgical_tiny_bert/pytorch_model.pt'
OUT_PT_INT8 = './models/surgical_tiny_bert_int8.pt'
OUT_ONNX_FP = './models/surgical_tiny_bert.onnx'
OUT_ONNX_Q8 = './models/surgical_tiny_bert_q8.onnx'
OUT_TOK     = './models/surgical_tiny_bert/tokenizer.json'
OUT_CFG     = './models/surgical_tiny_bert/config.json'

# -- TinySurgicalBERT architecture ---------------------------------------------
VOCAB_SIZE       = 4096    # Surgical BPE vocabulary (vs 28,996 for BERT)
D_MODEL          = 128     # Hidden dimension
N_HEAD           = 4       # Attention heads  (D_MODEL / N_HEAD = 32 per head)
NUM_LAYERS       = 2       # Transformer encoder layers
DIM_FEEDFORWARD  = 256     # FFN inner dimension
DROPOUT          = 0.1
MAX_SEQ_LEN      = 66      # 64 content tokens + [CLS] + [SEP]
OUTPUT_DIM       = 256     # Final embedding dim (> 200 so all FEATURES_PER_COL PCA targets work)

# -- BPE tokenizer -------------------------------------------------------------
TOK_VOCAB_SIZE   = 4096
TOK_MIN_FREQ     = 2

# -- Teacher PCA ---------------------------------------------------------------
TEACHER_PCA_DIM  = 256     # Teacher projected to same dim as student output

# -- Distillation training -----------------------------------------------------
DISTILL_EPOCHS      = 150
DISTILL_BATCH_SIZE  = 256
DISTILL_LR          = 3e-4
DISTILL_WEIGHT_DECAY= 1e-4
DISTILL_PATIENCE    = 15    # Early stopping patience (epochs)
DISTILL_LR_PATIENCE = 5     # LR reduction patience
DISTILL_LR_FACTOR   = 0.5
DISTILL_MIN_LR      = 1e-6
DISTILL_CLIP_NORM   = 1.0
VAL_FRACTION        = 0.1   # 10% of corpus for distillation validation
RANDOM_STATE        = 42
LOSS_ALPHA          = 0.5   # weight for MSE loss; (1-alpha) for cosine loss

# =============================================================================
# IMPORTS
# =============================================================================
import os, sys, json, sqlite3, time, warnings, struct, copy
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(BERT_DIR,   exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs('./models', exist_ok=True)

def _require(pkg, install_cmd):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"\n  ERROR: '{pkg}' not installed.\n  Fix : {install_cmd}\n")
        sys.exit(1)

torch = _require('torch', 'pip install torch')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_require('sklearn', 'pip install scikit-learn')
from sklearn.decomposition import PCA

_require('tokenizers', 'pip install tokenizers')
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.normalizers import Sequence as NormSeq
from tokenizers.processors import TemplateProcessing

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("  WARNING: onnx/onnxruntime not installed -- ONNX export disabled.")
    print("           Fix: pip install onnx onnxruntime")

# =============================================================================
# UTILITIES
# =============================================================================

def sep(title='', width=72, char='='):
    if title:
        print(f"\n{char*width}\n  {title}\n{char*width}")
    else:
        print(char * width)

def _file_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 ** 2)
    return 0.0

def _count_params(model):
    return sum(p.numel() for p in model.parameters())

def _set_seed(seed=RANDOM_STATE):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# STAGE D1 -- LOAD CORPUS + TEACHER EMBEDDINGS
# =============================================================================

def _d1_load_texts():
    if not os.path.exists(DB_PATH):
        print(f"\n  ERROR: {DB_PATH} not found.")
        print("  Run pipeline.py Stage 01 first.")
        sys.exit(1)
    with sqlite3.connect(DB_PATH) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if CLEAN_TABLE not in tables:
            print(f"  ERROR: Table '{CLEAN_TABLE}' missing in {DB_PATH}.")
            print("  Run pipeline.py Stage 01 first.")
            sys.exit(1)
        df = pd.read_sql(f"SELECT [{TEXT_COL}] FROM {CLEAN_TABLE}", conn)
    texts = df[TEXT_COL].astype(str).str.lower().str.strip().tolist()
    print(f"  Loaded {len(texts):,} surgical procedure texts from '{TEXT_COL}'")
    print(f"  Sample: {texts[:3]}")
    return texts

def _d1_load_teacher(texts):
    """Load teacher embeddings in priority order:
    1. ClinicalBERT cache (domain-specific, 768-d)
    2. SentenceBERT cache (general, 384-d)
    3. Compute SentenceBERT on-the-fly (fallback)
    """
    if os.path.exists(TEACHER_CLINICALBERT):
        arr = np.load(TEACHER_CLINICALBERT)
        print(f"  Teacher: Bio_ClinicalBERT  shape={arr.shape}  "
              f"source={TEACHER_CLINICALBERT}")
        teacher_name = 'Bio_ClinicalBERT'
        return arr.astype(np.float32), teacher_name

    if os.path.exists(TEACHER_SENTENCEBERT):
        arr = np.load(TEACHER_SENTENCEBERT)
        print(f"  Teacher: SentenceBERT  shape={arr.shape}  "
              f"source={TEACHER_SENTENCEBERT}")
        teacher_name = 'SentenceBERT (all-MiniLM-L6-v2)'
        return arr.astype(np.float32), teacher_name

    print(f"  No BERT cache found -- computing SentenceBERT on-the-fly...")
    print(f"  (Run pipeline.py Stage 02 first for faster startup)")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: sentence-transformers not installed.")
        print("  Fix: pip install sentence-transformers")
        sys.exit(1)
    model = SentenceTransformer(SENTENCEBERT_MODEL_ID)
    arr = model.encode(texts, show_progress_bar=True, batch_size=64)
    arr = arr.astype(np.float32)
    out = os.path.join(BERT_DIR, f'sentencebert_{TEXT_COL}.npy')
    np.save(out, arr)
    print(f"  Computed + cached SentenceBERT  shape={arr.shape}  -> {out}")
    teacher_name = 'SentenceBERT (all-MiniLM-L6-v2) [freshly computed]'
    return arr, teacher_name

def _d1_project_teacher(teacher_emb):
    """PCA-project teacher to OUTPUT_DIM for distillation targets."""
    dim = min(OUTPUT_DIM, teacher_emb.shape[1], teacher_emb.shape[0] - 1)
    if dim < OUTPUT_DIM:
        print(f"  NOTE: Reducing OUTPUT_DIM from {OUTPUT_DIM} to {dim} "
              f"(teacher_dim={teacher_emb.shape[1]}, n_texts={teacher_emb.shape[0]})")
    pca = PCA(n_components=dim, random_state=RANDOM_STATE)
    projected = pca.fit_transform(teacher_emb).astype(np.float32)
    var_exp = pca.explained_variance_ratio_.sum() * 100
    print(f"  Teacher PCA: {teacher_emb.shape[1]}-d -> {dim}-d  "
          f"variance_explained={var_exp:.1f}%")
    return projected, pca, dim

def run_stage_d1():
    sep("STAGE D1 -- LOAD CORPUS + TEACHER EMBEDDINGS")
    texts = _d1_load_texts()
    teacher_emb, teacher_name = _d1_load_teacher(texts)

    # Sanity checks
    assert len(texts) == len(teacher_emb), (
        f"Mismatch: {len(texts)} texts vs {len(teacher_emb)} embeddings. "
        "Re-run pipeline.py Stage 02."
    )
    nan_count = np.isnan(teacher_emb).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in teacher embeddings -- replacing with 0.")
        teacher_emb = np.nan_to_num(teacher_emb)

    projected, teacher_pca, actual_dim = _d1_project_teacher(teacher_emb)

    print(f"\n  Summary:")
    print(f"    Corpus size     : {len(texts):,} texts")
    print(f"    Teacher model   : {teacher_name}")
    print(f"    Teacher dim     : {teacher_emb.shape[1]}")
    print(f"    Distill targets : {actual_dim}-d  (PCA-projected teacher)")
    print(f"\n  [OK] Stage D1 complete.")
    return texts, teacher_emb, projected, teacher_pca, actual_dim, teacher_name

# =============================================================================
# STAGE D2 -- BUILD SURGICAL BPE TOKENIZER
# =============================================================================

def _d2_train_tokenizer(texts):
    """Train a BPE tokenizer on the surgical procedure corpus."""
    # Special tokens -- IDs are position in this list
    SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer  = NormSeq([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", CLS_ID), ("[SEP]", SEP_ID)],
    )

    trainer = BpeTrainer(
        vocab_size=TOK_VOCAB_SIZE,
        min_frequency=TOK_MIN_FREQ,
        special_tokens=SPECIAL,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.enable_padding(pad_id=PAD_ID, pad_token="[PAD]", length=MAX_SEQ_LEN)
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    return tok

def _d2_encode_batch(tok, texts):
    encodings = tok.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings],            dtype=np.int64)
    attn_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    return input_ids, attn_mask

def run_stage_d2(texts):
    sep("STAGE D2 -- BUILD SURGICAL BPE TOKENIZER")

    if os.path.exists(OUT_TOK):
        print(f"  Loading cached tokenizer from {OUT_TOK}")
        tok = Tokenizer.from_file(OUT_TOK)
        tok.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_SEQ_LEN)
        tok.enable_truncation(max_length=MAX_SEQ_LEN)
    else:
        print(f"  Training BPE tokenizer on {len(texts):,} texts ...")
        t0 = time.time()
        tok = _d2_train_tokenizer(texts)
        tok.save(OUT_TOK)
        print(f"  Trained in {time.time()-t0:.1f}s  -> {OUT_TOK}")

    vocab_size = tok.get_vocab_size()
    print(f"\n  Vocabulary size : {vocab_size:,}  (vs BERT: 28,996)")
    print(f"  Max seq length  : {MAX_SEQ_LEN}")

    # Vocabulary coverage check
    enc = _d2_encode_batch(tok, texts[:500])
    unk_id = tok.token_to_id("[UNK]")
    unk_rate = (enc[0] == unk_id).sum() / enc[0].size * 100
    print(f"  UNK rate (sample 500): {unk_rate:.2f}%  "
          f"({'good' if unk_rate < 2 else 'consider increasing VOCAB_SIZE'})")

    # Token length distribution
    lengths = (enc[1].sum(axis=1))
    print(f"  Token length (sample 500): "
          f"mean={lengths.mean():.1f}  "
          f"median={np.median(lengths):.0f}  "
          f"max={lengths.max()}")

    print(f"\n  [OK] Stage D2 complete.  Tokenizer -> {OUT_TOK}")
    return tok, vocab_size

# =============================================================================
# MODEL DEFINITION -- TinySurgicalBERT
# =============================================================================

class TinySurgicalBERT(nn.Module):
    """
    Lightweight transformer encoder for surgical procedure text.

    Architecture:
        BPE(vocab=4096) -> Embedding(128) + PosEmb(128)
        -> 2x TransformerEncoderLayer(d=128, heads=4, ffn=256, Pre-LN)
        -> LayerNorm -> CLS pooling -> Linear(128 -> output_dim)

    Parameters: ~1.5M  (~6 MB fp32 / ~1.5 MB INT8)
    vs Bio_ClinicalBERT: 110M params / 440 MB fp32
    """
    def __init__(self, vocab_size, d_model=D_MODEL, nhead=N_HEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
                 output_dim=OUTPUT_DIM):
        super().__init__()
        self.d_model    = d_model
        self.output_dim = output_dim

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)

        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids     : LongTensor [B, S]
            attention_mask: LongTensor [B, S]  (1=real, 0=pad)
        Returns:
            embeddings    : FloatTensor [B, output_dim]
        """
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.embed_norm(x)
        x = self.embed_drop(x)

        # PyTorch TransformerEncoder: key_padding_mask is True where PADDING
        key_padding_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        # CLS token pooling (position 0 = [CLS] inserted by post_processor)
        cls_emb = x[:, 0, :]
        return self.output_proj(cls_emb)

# =============================================================================
# STAGE D3 -- KNOWLEDGE DISTILLATION
# =============================================================================

def _distillation_loss(student, teacher, alpha=LOSS_ALPHA):
    """Combined MSE + cosine embedding loss (L2-normalised targets)."""
    s_norm = F.normalize(student, dim=-1)
    t_norm = F.normalize(teacher, dim=-1)
    mse  = F.mse_loss(s_norm, t_norm)
    cos  = (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    return alpha * mse + (1.0 - alpha) * cos

def _make_loaders(input_ids, attn_mask, targets, val_frac=VAL_FRACTION,
                  batch_size=DISTILL_BATCH_SIZE, seed=RANDOM_STATE):
    n = len(input_ids)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(n * val_frac))
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    def _ds(i):
        return TensorDataset(
            torch.tensor(input_ids[i], dtype=torch.long),
            torch.tensor(attn_mask[i], dtype=torch.long),
            torch.tensor(targets[i],   dtype=torch.float32),
        )
    # pin_memory=False: avoids CUDA memory pinning issues on some drivers
    tr_loader = DataLoader(_ds(tr_idx), batch_size=batch_size,
                           shuffle=True,  pin_memory=False, drop_last=False)
    va_loader = DataLoader(_ds(va_idx), batch_size=batch_size * 2,
                           shuffle=False, pin_memory=False)
    print(f"  Train: {len(tr_idx):,}  |  Val: {len(va_idx):,}  "
          f"(batch_size={batch_size})")
    return tr_loader, va_loader

def _eval_loss(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for ids, mask, tgt in loader:
            ids, mask, tgt = ids.to(device), mask.to(device), tgt.to(device)
            out = model(ids, mask)
            total_loss += _distillation_loss(out, tgt).item() * len(ids)
            n += len(ids)
    return total_loss / n

def run_stage_d3(texts, projected_targets, actual_dim, teacher_name,
                 tok, vocab_size):
    sep("STAGE D3 -- KNOWLEDGE DISTILLATION")

    if os.path.exists(OUT_PT_FP32) and os.path.exists(OUT_CFG):
        print(f"  Checkpoint already exists at {OUT_PT_FP32} -- skipping training.")
        print(f"  Delete {OUT_PT_FP32} to retrain.")
        cfg = json.load(open(OUT_CFG))
        model = TinySurgicalBERT(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            max_seq_len=cfg['max_seq_len'],
            output_dim=cfg['output_dim'],
        )
        model.load_state_dict(torch.load(OUT_PT_FP32, map_location='cpu',
                                         weights_only=True))
        return model

    _set_seed()
    device = _device()
    print(f"  Device: {device}")

    # Tokenize all texts
    print(f"\n  Tokenizing {len(texts):,} texts ...")
    t0 = time.time()
    input_ids, attn_mask = _d2_encode_batch(tok, texts)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"shape=({input_ids.shape[0]}, {input_ids.shape[1]})")

    # Data loaders
    tr_loader, va_loader = _make_loaders(input_ids, attn_mask, projected_targets)

    # Model
    model = TinySurgicalBERT(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        output_dim=actual_dim,
    ).to(device)

    n_params = _count_params(model)
    print(f"\n  TinySurgicalBERT parameters : {n_params:,}  "
          f"(~{n_params*4/1e6:.1f} MB fp32 / "
          f"~{n_params/1e6:.1f} MB INT8)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DISTILL_LR,
        weight_decay=DISTILL_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=DISTILL_LR_FACTOR,
        patience=DISTILL_LR_PATIENCE, min_lr=DISTILL_MIN_LR,
    )

    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0
    CKPT_PATH      = os.path.join(MODEL_DIR, '_checkpoint.pt')
    CKPT_META      = os.path.join(MODEL_DIR, '_checkpoint_meta.json')

    # Resume from intermediate checkpoint if available (crash recovery)
    start_epoch = 1
    if os.path.exists(CKPT_PATH) and os.path.exists(CKPT_META):
        try:
            meta = json.load(open(CKPT_META))
            model.load_state_dict(torch.load(CKPT_PATH, map_location=device,
                                             weights_only=True))
            best_val_loss  = meta['best_val_loss']
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = meta['patience_count']
            start_epoch    = meta['epoch'] + 1
            print(f"  Resuming from epoch {meta['epoch']}  "
                  f"(best_val={best_val_loss:.5f})")
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint ({e}) — starting fresh.")
            start_epoch = 1

    print(f"\n  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'LR':>12} {'Patience':>10}")
    print(f"  {'-'*56}")

    for epoch in range(start_epoch, DISTILL_EPOCHS + 1):
        # -- train -----------------------------------------------------------
        model.train()
        tr_loss, n_tr = 0.0, 0
        cuda_error = False
        for ids, mask, tgt in tr_loader:
            try:
                ids, mask, tgt = ids.to(device), mask.to(device), tgt.to(device)
                optimizer.zero_grad()
                out  = model(ids, mask)
                loss = _distillation_loss(out, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), DISTILL_CLIP_NORM)
                optimizer.step()
                tr_loss += loss.item() * len(ids)
                n_tr    += len(ids)
            except RuntimeError as e:
                if 'cuda' in str(e).lower() or 'CUDA' in str(e):
                    print(f"\n  WARNING: CUDA error at epoch {epoch}: {e}")
                    print(f"  Switching to CPU and resuming ...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    device = torch.device('cpu')
                    model  = model.to(device)
                    cuda_error = True
                    break
                raise
        if cuda_error:
            # Redo this epoch on CPU
            tr_loss, n_tr = 0.0, 0
            for ids, mask, tgt in tr_loader:
                ids, mask, tgt = ids.to(device), mask.to(device), tgt.to(device)
                optimizer.zero_grad()
                out  = model(ids, mask)
                loss = _distillation_loss(out, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), DISTILL_CLIP_NORM)
                optimizer.step()
                tr_loss += loss.item() * len(ids)
                n_tr    += len(ids)
        if n_tr == 0:
            continue
        tr_loss /= n_tr

        # -- validate --------------------------------------------------------
        val_loss = _eval_loss(model, va_loader, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch <= 5 or patience_count == 0:
            flag = ' <-best' if patience_count == 0 else ''
            print(f"  {epoch:>6} {tr_loss:>12.5f} {val_loss:>12.5f} "
                  f"{current_lr:>12.2e} {patience_count:>10}{flag}")

        # Save checkpoint every 5 epochs for crash recovery
        if epoch % 5 == 0 and best_state is not None:
            torch.save(best_state, CKPT_PATH)
            json.dump({'epoch': epoch, 'best_val_loss': float(best_val_loss),
                       'patience_count': patience_count},
                      open(CKPT_META, 'w'))

        if patience_count >= DISTILL_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {DISTILL_PATIENCE} epochs)")
            break

    # Restore best
    model.load_state_dict(best_state)
    model.eval()
    print(f"\n  Best val loss : {best_val_loss:.5f}")

    # Clean up checkpoint files
    for p in [CKPT_PATH, CKPT_META]:
        if os.path.exists(p):
            os.remove(p)

    # Save checkpoint
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), OUT_PT_FP32)

    # Save config
    cfg = {
        'vocab_size': vocab_size,
        'd_model': D_MODEL,
        'nhead': N_HEAD,
        'num_layers': NUM_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'max_seq_len': MAX_SEQ_LEN,
        'output_dim': actual_dim,
        'teacher_model': teacher_name,
        'teacher_pca_dim': actual_dim,
        'text_column': TEXT_COL,
        'n_params': n_params,
        'best_val_loss': float(best_val_loss),
    }
    with open(OUT_CFG, 'w') as f:
        json.dump(cfg, f, indent=2)

    size_mb = _file_mb(OUT_PT_FP32)
    print(f"\n  [OK] Stage D3 complete.")
    print(f"     Model  -> {OUT_PT_FP32}  ({size_mb:.1f} MB)")
    print(f"     Config -> {OUT_CFG}")
    return model_cpu

# =============================================================================
# STAGE D4 -- INT8 QUANTISATION + ONNX EXPORT
# =============================================================================

def _build_dummy_inputs(tok, n=2):
    """Create dummy tokenised inputs for ONNX tracing."""
    samples = ["total knee arthroplasty bilateral",
               "laparoscopic appendectomy"][:n]
    ids, mask = _d2_encode_batch(tok, samples)
    return (
        torch.tensor(ids,  dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )

def run_stage_d4(model, tok, actual_dim):
    sep("STAGE D4 -- INT8 QUANTISATION + ONNX EXPORT")
    model.eval()

    # -- 1. PyTorch dynamic INT8 quantisation ---------------------------------
    # Note: nn.Embedding requires float_qparams_weight_only_qconfig, not the
    # default dynamic qconfig. We quantize only the Linear layers here; the
    # ONNX INT8 quantisation in step 4.3 covers the full model uniformly.
    print("  [4.1]  PyTorch dynamic INT8 quantisation (Linear layers) ...")
    model_int8 = torch.quantization.quantize_dynamic(
        copy.deepcopy(model),
        {nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(model_int8.state_dict(), OUT_PT_INT8)
    size_fp32 = _file_mb(OUT_PT_FP32)
    size_int8 = _file_mb(OUT_PT_INT8)
    print(f"  fp32 : {size_fp32:.2f} MB")
    print(f"  INT8 : {size_int8:.2f} MB  "
          f"({size_fp32/size_int8:.1f}x smaller)")

    if not ONNX_AVAILABLE:
        print("\n  Skipping ONNX export (onnx/onnxruntime not installed).")
        print(f"\n  [OK] Stage D4 complete (PyTorch INT8 only).")
        return

    # -- 2. ONNX export (fp32) -------------------------------------------------
    print("\n  [4.2]  ONNX export ...")
    dummy_ids, dummy_mask = _build_dummy_inputs(tok)

    try:
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            OUT_ONNX_FP,
            opset_version=14,
            input_names=['input_ids', 'attention_mask'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids':      {0: 'batch', 1: 'seq'},
                'attention_mask': {0: 'batch', 1: 'seq'},
                'embeddings':     {0: 'batch'},
            },
            do_constant_folding=True,
        )
        onnx_model = onnx.load(OUT_ONNX_FP)
        onnx.checker.check_model(onnx_model)
        size_onnx_fp = _file_mb(OUT_ONNX_FP)
        print(f"  ONNX fp32 : {size_onnx_fp:.2f} MB  -> {OUT_ONNX_FP}")
    except Exception as e:
        print(f"  WARNING: ONNX export failed: {e}")
        print(f"  Continuing without ONNX output.")
        return

    # -- 3. ONNX INT8 quantisation ---------------------------------------------
    print("\n  [4.3]  ONNX dynamic INT8 quantisation ...")
    try:
        quantize_dynamic(
            model_input=OUT_ONNX_FP,
            model_output=OUT_ONNX_Q8,
            weight_type=QuantType.QUInt8,
        )
        size_onnx_q8 = _file_mb(OUT_ONNX_Q8)
        print(f"  ONNX INT8 : {size_onnx_q8:.2f} MB  -> {OUT_ONNX_Q8}")
        reduction = size_fp32 / size_onnx_q8 if size_onnx_q8 > 0 else 0
        print(f"  Total reduction vs Bio_ClinicalBERT (~440 MB): "
              f"{440/size_onnx_q8:.0f}x  smaller")
    except Exception as e:
        print(f"  WARNING: ONNX quantisation failed: {e}")

    # -- 4. Validate ONNX output matches PyTorch -------------------------------
    print("\n  [4.4]  Validating ONNX vs PyTorch outputs ...")
    try:
        sess = ort.InferenceSession(OUT_ONNX_FP,
                                    providers=['CPUExecutionProvider'])
        onnx_out = sess.run(
            None,
            {'input_ids': dummy_ids.numpy(),
             'attention_mask': dummy_mask.numpy()}
        )[0]
        with torch.no_grad():
            pt_out = model(dummy_ids, dummy_mask).numpy()
        max_diff = np.abs(pt_out - onnx_out).max()
        print(f"  Max abs difference (ONNX vs PyTorch) : {max_diff:.2e}  "
              f"({'[OK] OK' if max_diff < 1e-4 else '[!] check'})")
    except Exception as e:
        print(f"  WARNING: ONNX validation failed: {e}")

    # -- 5. Latency benchmark --------------------------------------------------
    print("\n  [4.5]  Latency benchmark (batch=1, CPU) ...")
    try:
        ids_1 = dummy_ids[:1].numpy()
        msk_1 = dummy_mask[:1].numpy()
        sess = ort.InferenceSession(OUT_ONNX_Q8,
                                    providers=['CPUExecutionProvider'])
        # Warmup
        for _ in range(10):
            sess.run(None, {'input_ids': ids_1, 'attention_mask': msk_1})
        # Measure
        n_runs = 200
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {'input_ids': ids_1, 'attention_mask': msk_1})
        ms_per_sample = (time.perf_counter() - t0) / n_runs * 1000
        print(f"  ONNX INT8 latency : {ms_per_sample:.2f} ms/sample  "
              f"(batch=1, CPU)")
        print(f"  Throughput        : {1000/ms_per_sample:.0f} samples/s")
    except Exception as e:
        print(f"  WARNING: Benchmark failed: {e}")

    print(f"\n  [OK] Stage D4 complete.")

# =============================================================================
# STAGE D5 -- GENERATE EMBEDDINGS + VALIDATE + SAVE CACHE
# =============================================================================

def _embed_all(model, tok, texts, batch_size=512, device=None):
    """Run all texts through the student model, return (N, output_dim) array."""
    if device is None:
        device = _device()
    model = model.to(device).eval()

    input_ids, attn_mask = _d2_encode_batch(tok, texts)
    all_emb = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            ids  = torch.tensor(input_ids[i:i+batch_size], dtype=torch.long).to(device)
            mask = torch.tensor(attn_mask[i:i+batch_size], dtype=torch.long).to(device)
            emb  = model(ids, mask).cpu().numpy()
            all_emb.append(emb)
            if (i // batch_size) % 20 == 0:
                print(f"    Batch {i//batch_size+1}/{n_batches}  "
                      f"({i:,}/{len(texts):,})")
    return np.vstack(all_emb).astype(np.float32)

def _pearson_r(a, b):
    """Mean Pearson R across matched dimensions (both (N, D) arrays)."""
    rs = []
    n = min(a.shape[1], b.shape[1])
    for d in range(n):
        x, y = a[:, d], b[:, d]
        r = np.corrcoef(x, y)[0, 1]
        if not np.isnan(r):
            rs.append(abs(r))
    return np.mean(rs) if rs else 0.0

def run_stage_d5(model, tok, texts, teacher_emb, projected_targets,
                 teacher_pca, actual_dim, teacher_name):
    sep("STAGE D5 -- GENERATE EMBEDDINGS + VALIDATE + SAVE CACHE")

    # -- Generate student embeddings --------------------------------------------
    print(f"  Generating TinySurgicalBERT embeddings for {len(texts):,} texts ...")
    t0 = time.time()
    student_emb = _embed_all(model, tok, texts)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  "
          f"({len(texts)/elapsed:.0f} samples/s)  shape={student_emb.shape}")

    # -- Validate: correlation with projected teacher ---------------------------
    print("\n  Validation -- comparing student to projected teacher:")
    # Cosine similarity
    s_norm = student_emb / (np.linalg.norm(student_emb, axis=1, keepdims=True) + 1e-9)
    t_norm = projected_targets / (np.linalg.norm(projected_targets, axis=1, keepdims=True) + 1e-9)
    cos_sim = (s_norm * t_norm).sum(axis=1)
    print(f"    Cosine similarity  : mean={cos_sim.mean():.3f}  "
          f"median={np.median(cos_sim):.3f}  "
          f"p10={np.percentile(cos_sim, 10):.3f}")

    # Pearson R
    r_val = _pearson_r(student_emb, projected_targets)
    print(f"    Mean abs Pearson R : {r_val:.3f}  "
          f"({'excellent' if r_val > 0.85 else 'good' if r_val > 0.70 else 'acceptable'})")

    # MSE
    mse = np.mean((student_emb - projected_targets) ** 2)
    print(f"    MSE                : {mse:.5f}")

    # -- Save pipeline-compatible cache ----------------------------------------
    np.save(OUT_NPY, student_emb)
    size_npy_mb = _file_mb(OUT_NPY)
    print(f"\n  Pipeline cache saved -> {OUT_NPY}")
    print(f"    Shape  : {student_emb.shape}")
    print(f"    Size   : {size_npy_mb:.1f} MB")

    # -- Size comparison table --------------------------------------------------
    print("\n" + "="*72)
    print("  SIZE COMPARISON SUMMARY")
    print("="*72)
    print(f"  {'Model':<38} {'Params':>10} {'Size':>10} {'Ratio':>8}")
    print(f"  {'-'*68}")

    rows = [
        ("Bio_ClinicalBERT (teacher)",  "110 M",  "~440 MB",  "1x"),
        ("all-MiniLM-L6-v2 (SentBERT)", "22 M",   " ~90 MB",  "5x"),
        ("TinySurgicalBERT fp32",        f"{_count_params(model)/1e6:.1f} M",
         f"{_file_mb(OUT_PT_FP32):.1f} MB",
         f"{440/_file_mb(OUT_PT_FP32):.0f}x"),
        ("TinySurgicalBERT INT8 (PyTorch)", "--",
         f"{_file_mb(OUT_PT_INT8):.1f} MB",
         f"{440/_file_mb(OUT_PT_INT8):.0f}x"),
    ]
    if ONNX_AVAILABLE and os.path.exists(OUT_ONNX_FP):
        rows.append(("TinySurgicalBERT ONNX fp32", "--",
                     f"{_file_mb(OUT_ONNX_FP):.1f} MB",
                     f"{440/_file_mb(OUT_ONNX_FP):.0f}x"))
    if ONNX_AVAILABLE and os.path.exists(OUT_ONNX_Q8):
        rows.append(("TinySurgicalBERT ONNX INT8  <- mobile", "--",
                     f"{_file_mb(OUT_ONNX_Q8):.1f} MB",
                     f"{440/_file_mb(OUT_ONNX_Q8):.0f}x"))

    for name, params, size, ratio in rows:
        print(f"  {name:<38} {params:>10} {size:>10} {ratio:>8}")

    # -- Integration instructions -----------------------------------------------
    print("\n" + "="*72)
    print("  PIPELINE INTEGRATION INSTRUCTIONS")
    print("="*72)
    print("""
  To use TinySurgicalBERT in pipeline.py:

  1. Stage 02 -- add to S02_TASKS:
       3: ('tinybert', 'tinybert_scheduled_procedure.npy')
     (Cache already generated -- Stage 02 will skip if file exists.)

  2. Stage 03 -- add to BERT_ENCODINGS:
       BERT_ENCODINGS = ['clinicalbert', 'sentencebert', 'tinybert']

  3. Stage 03 -- tinybert output is 256-d, so update FEATURES_PER_COL if
     you only want tinybert: [10, 50, 100, 200]  (all valid since 256 > 200)

  4. For new mobile inference (replacing Stage 02 on-device):
       * Tokenizer  : ./models/surgical_tiny_bert/tokenizer.json
       * ONNX model : ./models/surgical_tiny_bert_q8.onnx
       * Input  : input_ids [1, 66], attention_mask [1, 66]  (int64)
       * Output : embeddings [1, 256]  (float32)

  Mobile runtime requirement: ONNX Runtime Mobile (~3 MB) or
  CoreML (iOS) / TensorFlow Lite converted from ONNX.
""")

    print(f"  [OK] Stage D5 complete.")
    print(f"\n  All outputs in ./models/  and  {BERT_DIR}/")

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep("TinySurgicalBERT -- Knowledge Distillation Pipeline", char='#')
    print(f"\n  Goal    : {OUTPUT_DIM}-d embeddings from {D_MODEL}-d model")
    print(f"  Target  : < 6 MB fp32 / < 2 MB INT8  (mobile-deployable)")
    print(f"  Teacher : Bio_ClinicalBERT (preferred) or SentenceBERT")
    print(f"  Column  : {TEXT_COL}  (preoperative text -- available before surgery)")

    t_total = time.time()

    # -- D1: Load corpus + teacher ----------------------------------------------
    texts, teacher_emb, projected, teacher_pca, actual_dim, teacher_name = \
        run_stage_d1()

    # -- D2: Surgical BPE tokenizer ---------------------------------------------
    tok, vocab_size = run_stage_d2(texts)

    # -- D3: Knowledge distillation ---------------------------------------------
    model = run_stage_d3(
        texts, projected, actual_dim, teacher_name, tok, vocab_size
    )

    # -- D4: Quantisation + ONNX ------------------------------------------------
    run_stage_d4(model, tok, actual_dim)

    # -- D5: Validate + save cache ----------------------------------------------
    run_stage_d5(
        model, tok, texts, teacher_emb, projected,
        teacher_pca, actual_dim, teacher_name
    )

    elapsed = (time.time() - t_total) / 60
    sep(f"COMPLETE  ({elapsed:.1f} min total)", char='#')


if __name__ == '__main__':
    main()
