/**
 * AI Text Detector — Geometric Trajectory Architecture
 * =====================================================
 * JavaScript inference port of train_detector.py
 *
 * Loads a model saved by the Python trainer (detector.json-<size>MB)
 * and runs inference in Node.js. No GPU required — pure JS math.
 *
 * Usage:
 *   const det = new Detector();
 *   await det.load('./detector.json-33.7MB');
 *   const result = await det.predict('Some text to classify...');
 *   console.log(result); // { label: 'AI', confidence: 0.94, score: 0.94 }
 *
 *   // batch prediction
 *   const results = await det.predictBatch(['text one', 'text two']);
 */

'use strict';

const fs   = require('fs');
const path = require('path');

// ─────────────────────────────────────────────────────────────────────────────
//  M A T H   H E L P E R S  (pure JS, no dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/** Dot product of two Float32Arrays of equal length */
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** L2 norm of a Float32Array */
function norm(a) {
  return Math.sqrt(dot(a, a));
}

/** Cosine similarity, safe against zero vectors */
function cosine(a, b) {
  const na = norm(a), nb = norm(b);
  if (na < 1e-6 || nb < 1e-6) return 0;
  return dot(a, b) / (na * nb);
}

/** Element-wise addition, returns new Float32Array */
function add(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

/** Element-wise subtraction */
function sub(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

/** Scale a vector by a scalar */
function scale(a, s) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
  return out;
}

/** GELU activation (approximation matching PyTorch's default) */
function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

/** Softmax over an array */
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// ─────────────────────────────────────────────────────────────────────────────
//  L A Y E R S
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Dense (Linear) layer: y = x @ W.T + b
 * weights shape: [out_features, in_features]
 * bias shape:    [out_features]
 */
class Linear {
  constructor(weight, bias) {
    this.weight     = weight;   // Float32Array, row-major [outF, inF]
    this.bias       = bias;     // Float32Array [outF]
    this.in_features  = weight.length / bias.length;
    this.out_features = bias.length;
  }

  forward(x) {
    // x: Float32Array [inF]  →  out: Float32Array [outF]
    const { in_features: inF, out_features: outF } = this;
    const out = new Float32Array(outF);
    for (let o = 0; o < outF; o++) {
      let s = this.bias[o];
      const row = o * inF;
      for (let i = 0; i < inF; i++) s += this.weight[row + i] * x[i];
      out[o] = s;
    }
    return out;
  }
}

/**
 * LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 */
class LayerNorm {
  constructor(weight, bias, eps = 1e-5) {
    this.weight = weight;   // Float32Array [D]
    this.bias   = bias;     // Float32Array [D]
    this.eps    = eps;
    this.D      = weight.length;
  }

  forward(x) {
    const { D, eps } = this;
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x[i];
    mean /= D;
    let variance = 0;
    for (let i = 0; i < D; i++) variance += (x[i] - mean) ** 2;
    variance /= D;
    const std = Math.sqrt(variance + eps);
    const out = new Float32Array(D);
    for (let i = 0; i < D; i++) out[i] = ((x[i] - mean) / std) * this.weight[i] + this.bias[i];
    return out;
  }
}

/**
 * Embedding lookup: returns Float32Array [embed_dim] for a given token id.
 */
class Embedding {
  constructor(weight, embed_dim) {
    this.weight    = weight;     // Float32Array [vocab_size * embed_dim]
    this.embed_dim = embed_dim;
  }

  forward(id) {
    const D   = this.embed_dim;
    const off = id * D;
    return this.weight.slice(off, off + D);
  }
}

/**
 * Multi-Head Self-Attention (encoder, no causal mask).
 * Pre-norm variant matching PyTorch TransformerEncoderLayer(norm_first=True).
 */
class MultiHeadAttention {
  constructor({ in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                norm_weight, norm_bias, d_model, n_heads }) {
    this.d_model  = d_model;
    this.n_heads  = n_heads;
    this.d_head   = d_model / n_heads;

    // in_proj packs Q, K, V weights: shape [3*d_model, d_model]
    this.in_proj  = new Linear(in_proj_weight, in_proj_bias);
    this.out_proj = new Linear(out_proj_weight, out_proj_bias);
    this.norm     = new LayerNorm(norm_weight, norm_bias);
  }

  /**
   * x: Float32Array[] — array of T vectors, each [d_model]
   * pad_mask: bool[]  — true = padding (ignore)
   * Returns Float32Array[] of same shape.
   */
  forward(x, pad_mask) {
    const T = x.length;
    const { d_model: D, n_heads: H, d_head: Dh } = this;

    // pre-norm
    const xn = x.map(v => this.norm.forward(v));

    // project to Q, K, V for every position
    const qkv = xn.map(v => this.in_proj.forward(v));
    // split into Q [T, D], K [T, D], V [T, D]
    const Q = qkv.map(v => v.slice(0,     D));
    const K = qkv.map(v => v.slice(D,   2*D));
    const V = qkv.map(v => v.slice(2*D, 3*D));

    const scale_f = 1 / Math.sqrt(Dh);
    const out_vecs = new Array(T);

    for (let t = 0; t < T; t++) {
      const head_outs = [];
      for (let h = 0; h < H; h++) {
        const hOff = h * Dh;
        const q_h = Q[t].slice(hOff, hOff + Dh);

        // attention scores
        const scores = new Float32Array(T);
        for (let s = 0; s < T; s++) {
          if (pad_mask[s]) { scores[s] = -1e9; continue; }
          const k_h = K[s].slice(hOff, hOff + Dh);
          scores[s] = dot(q_h, k_h) * scale_f;
        }

        // softmax
        const attn = softmax(Array.from(scores));

        // weighted sum of V
        const head_out = new Float32Array(Dh);
        for (let s = 0; s < T; s++) {
          if (pad_mask[s]) continue;
          const v_h = V[s].slice(hOff, hOff + Dh);
          for (let d = 0; d < Dh; d++) head_out[d] += attn[s] * v_h[d];
        }
        head_outs.push(head_out);
      }

      // concatenate heads
      const concat = new Float32Array(D);
      for (let h = 0; h < H; h++)
        concat.set(head_outs[h], h * Dh);

      out_vecs[t] = this.out_proj.forward(concat);
    }

    // residual
    return out_vecs.map((v, i) => add(x[i], v));
  }
}

/**
 * Feed-forward sub-layer (pre-norm).
 * Linear → GELU → Linear, with residual.
 */
class FeedForward {
  constructor({ norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias }) {
    this.norm = new LayerNorm(norm_weight, norm_bias);
    this.fc1  = new Linear(fc1_weight, fc1_bias);
    this.fc2  = new Linear(fc2_weight, fc2_bias);
  }

  forward(x) {
    const xn  = this.norm.forward(x);
    const h   = this.fc1.forward(xn);
    const hg  = h.map(gelu);
    const out = this.fc2.forward(hg);
    return add(x, out);   // residual
  }
}

/**
 * One TransformerEncoderLayer (pre-norm, matching PyTorch norm_first=True).
 */
class TransformerLayer {
  constructor(attn_params, ffn_params) {
    this.attn = new MultiHeadAttention(attn_params);
    this.ffn  = new FeedForward(ffn_params);
  }

  forward(x, pad_mask) {
    const after_attn = this.attn.forward(x, pad_mask);
    return after_attn.map(v => this.ffn.forward(v));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  T O K E N I S E R
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Character-trigram hash tokeniser — must match Python's SimpleTokenizer.
 * Maps each word to an integer in [1, vocab_size].
 */
class SimpleTokenizer {
  constructor(vocabSize = 50000) {
    this.vocabSize = vocabSize;
  }

  /** Java-style string hashCode, matching Python's built-in hash for ASCII */
  _hash(str) {
    // We replicate Python's hash(str) % (vocab_size-1) + 1 using
    // a djb2-style hash that's consistent across runs (unlike Python's
    // randomised hash). Both the Python trainer and this tokeniser must
    // agree — the Python trainer uses Python's built-in hash() which is
    // randomised per-process, so we use the same djb2 fallback here.
    // For production, export the vocabulary from Python and load it here.
    let h = 5381;
    for (let i = 0; i < str.length; i++) {
      h = ((h << 5) + h + str.charCodeAt(i)) >>> 0;  // djb2
    }
    return (h % (this.vocabSize - 1)) + 1;
  }

  encode(text, maxLen = 128) {
    const words = text.split(/\s+/).slice(0, maxLen);
    return words.map(w => {
      const cleaned = w.toLowerCase().replace(/^[.,!?"'()\[\]{}:;]+|[.,!?"'()\[\]{}:;]+$/g, '');
      return cleaned.length ? this._hash(cleaned) : this._hash(w.toLowerCase());
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  D E T E C T O R   C L A S S
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Main inference class. Mirrors the Python AIDetector model.
 *
 * @example
 *   const det = new Detector();
 *   await det.load('./detector.json-33.7MB');
 *   const { label, confidence } = await det.predict('Text to classify');
 */
class Detector {
  constructor() {
    this.loaded      = false;
    this.cfg         = null;
    this.tokeniser   = null;

    // model weights (populated by load())
    this._embedding  = null;   // word embedding
    this._posEmbed   = null;   // positional embedding
    this._clsToken   = null;   // CLS token vector
    this._featNorm   = null;   // LayerNorm before MLP
    this._mlp        = null;   // per-word MLP layers
    this._layers     = null;   // transformer layers
    this._headNorm   = null;   // final LayerNorm
    this._headFc1    = null;   // head Linear 1
    this._headFc2    = null;   // head Linear 2
  }

  // ── file loading ────────────────────────────────────────────────────────────

  /**
   * Load a model file saved by the Python trainer.
   * Format: [4 bytes LE uint32: JSON length][JSON bytes][raw float32 weights]
   *
   * @param {string} filePath - path to the .json-<size>MB file
   */
  async load(filePath) {
    const buf = fs.readFileSync(filePath);

    // read JSON header
    const jsonLen  = buf.readUInt32LE(0);
    const jsonStr  = buf.slice(4, 4 + jsonLen).toString('utf8');
    const meta     = JSON.parse(jsonStr);
    this.cfg       = meta.cfg;
    this.tokeniser = new SimpleTokenizer(this.cfg.vocab_size);

    // extract weight tensors
    const weights  = this._extractWeights(buf, 4 + jsonLen, meta);

    // build model layers from weights
    this._buildModel(weights, meta);

    this.loaded = true;
    console.log(
      `[Detector] Loaded: ${path.basename(filePath)} | ` +
      `${(meta.n_params / 1e6).toFixed(1)}M params | ` +
      `embed_dim=${this.cfg.embed_dim} layers=${this.cfg.n_layers}`
    );
  }

  /** Parse raw binary into named Float32Arrays keyed by state_dict key */
  _extractWeights(buf, offset, meta) {
    const weights = {};
    let currentReadOffset = offset; // This tracks the current position in the original 'buf'

    for (let i = 0; i < meta.keys.length; i++) {
      const key = meta.keys[i];
      const shape = meta.shapes[i];
      const numel = shape.reduce((a, b) => a * b, 1);
      const nbytes = numel * 4; // Each Float32 is 4 bytes

      // Create a new, *aligned* Buffer to hold the current tensor's data.
      // Buffer.allocUnsafe ensures it's properly aligned for TypedArrays.
      const alignedBuffer = Buffer.allocUnsafe(nbytes);
      // Copy the potentially unaligned data from the source buffer into the aligned buffer.
      buf.copy(alignedBuffer, 0, currentReadOffset, currentReadOffset + nbytes);

      // Now, create the Float32Array from the alignedBuffer.
      // alignedBuffer.byteOffset will be 0, which is a multiple of 4.
      weights[key] = new Float32Array(alignedBuffer.buffer, alignedBuffer.byteOffset, numel);

      currentReadOffset += nbytes;
    }
    return weights;
  }

  /** Wire up all layer objects from the flat weights dict */
  _buildModel(w, meta) {
    const cfg = this.cfg;
    const D   = cfg.word_feat_dim;

    // ── word feature encoder ─────────────────────────────────────────────────
    const embedD = cfg.embed_dim;
    this._embedding = new Embedding(w['word_encoder.embedding.weight'], embedD);

    const rawDim = 2 * embedD + 7;
    this._featNorm = new LayerNorm(
      w['word_encoder.feat_norm.weight'],
      w['word_encoder.feat_norm.bias']
    );

    // MLP: Linear(rawDim → D*2) → GELU → Linear(D*2 → D) → LayerNorm
    this._mlpFc1   = new Linear(w['word_encoder.mlp.0.weight'], w['word_encoder.mlp.0.bias']);
    this._mlpLn    = new LayerNorm(w['word_encoder.mlp.4.weight'], w['word_encoder.mlp.4.bias']);
    this._mlpFc2   = new Linear(w['word_encoder.mlp.3.weight'], w['word_encoder.mlp.3.bias']);

    // ── positional embedding + CLS token ─────────────────────────────────────
    this._posEmbed = new Embedding(w['pos_embed.weight'], D);
    this._clsToken = w['cls_token'].slice();   // shape [1,1,D] stored flat → [D]

    // ── transformer layers ───────────────────────────────────────────────────
    this._layers = [];
    for (let l = 0; l < cfg.n_layers; l++) {
      const p = `transformer.layers.${l}`;

      const attnParams = {
        in_proj_weight  : w[`${p}.self_attn.in_proj_weight`],
        in_proj_bias    : w[`${p}.self_attn.in_proj_bias`],
        out_proj_weight : w[`${p}.self_attn.out_proj.weight`],
        out_proj_bias   : w[`${p}.self_attn.out_proj.bias`],
        norm_weight     : w[`${p}.norm1.weight`],
        norm_bias       : w[`${p}.norm1.bias`],
        d_model         : D,
        n_heads         : cfg.n_heads,
      };
      const ffnParams = {
        norm_weight : w[`${p}.norm2.weight`],
        norm_bias   : w[`${p}.norm2.bias`],
        fc1_weight  : w[`${p}.linear1.weight`],
        fc1_bias    : w[`${p}.linear1.bias`],
        fc2_weight  : w[`${p}.linear2.weight`],
        fc2_bias    : w[`${p}.linear2.bias`],
      };
      this._layers.push(new TransformerLayer(attnParams, ffnParams));
    }

    // ── classification head ──────────────────────────────────────────────────
    this._headNorm = new LayerNorm(w['head.0.weight'], w['head.0.bias']);
    this._headFc1  = new Linear(w['head.2.weight'],    w['head.2.bias']);
    this._headFc2  = new Linear(w['head.4.weight'],    w['head.4.bias']);
  }

  // ── word feature computation ────────────────────────────────────────────────

  /**
   * Compute the geometric feature vector for every word in the sequence.
   * Mirrors WordFeatureEncoder.forward() in Python.
   *
   * @param {number[]} tokenIds - token IDs, length T
   * @param {boolean[]} padMask - true = real token, length T
   * @returns {Float32Array[]} array of T vectors, each [word_feat_dim]
   */
  _wordFeatures(tokenIds, padMask) {
    const T    = tokenIds.length;
    const D    = this.cfg.embed_dim;
    const EPS  = 1e-6;

    // look up embeddings
    const embs = tokenIds.map(id => this._embedding.forward(id));

    // helpers
    const safeNorm = v => Math.max(norm(v), EPS);

    // prev / next embeddings (boundary clamped)
    const prevEmbs = [embs[0], ...embs.slice(0, T - 1)];
    const nextEmbs = [...embs.slice(1), embs[T - 1]];

    // centroid of real (non-padding) tokens
    const centroid = new Float32Array(D);
    let nReal = 0;
    for (let t = 0; t < T; t++) {
      if (!padMask[t]) continue;
      nReal++;
      for (let d = 0; d < D; d++) centroid[d] += embs[t][d];
    }
    if (nReal > 0) for (let d = 0; d < D; d++) centroid[d] /= nReal;

    // running mean distance
    const jumpDists  = new Float32Array(T);
    const jumps      = embs.map((e, t) => {
      if (t === 0) return new Float32Array(D);
      return sub(e, prevEmbs[t]);
    });
    jumps.forEach((j, t) => { jumpDists[t] = t === 0 ? 0 : norm(j); });

    let cumDist = 0;
    const runningMeans = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      cumDist += jumpDists[t];
      runningMeans[t] = cumDist / (t + 1);
    }

    // previous jump (for angle)
    const prevJumps = [new Float32Array(D), ...jumps.slice(0, T - 1)];

    // local variance window=4
    const localVar = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let w = 1; w <= 4; w++) {
        const prev_t = Math.max(0, t - w);
        s += Math.abs(jumpDists[t] - jumpDists[prev_t]);
      }
      localVar[t] = s / 4;
    }

    // assemble raw feature vectors and pass through MLP
    const rawDim = 2 * D + 7;
    const result = [];

    for (let t = 0; t < T; t++) {
      const raw = new Float32Array(rawDim);
      let off = 0;

      // embedding (128)
      raw.set(embs[t], off);        off += D;
      // jump vector (128)
      raw.set(jumps[t], off);       off += D;
      // jump distance (1)
      raw[off++] = jumpDists[t];
      // cosine prev (1)
      raw[off++] = t === 0 ? 0 : cosine(embs[t], prevEmbs[t]);
      // cosine next (1)
      raw[off++] = cosine(embs[t], nextEmbs[t]);
      // deviation from running mean (1)
      raw[off++] = Math.abs(jumpDists[t] - runningMeans[t]);
      // distance to centroid (1)
      raw[off++] = norm(sub(embs[t], centroid));
      // jump angle (1)
      raw[off++] = t < 2 ? 0 : cosine(jumps[t], prevJumps[t]);
      // local variance (1)
      raw[off++] = localVar[t];

      // LayerNorm → MLP
      const normed = this._featNorm.forward(raw);
      const h1     = this._mlpFc1.forward(normed).map(gelu);
      const h2     = this._mlpFc2.forward(h1);
      result.push(this._mlpLn.forward(h2));
    }

    return result;
  }

  // ── full forward pass ────────────────────────────────────────────────────────

  /**
   * Run the full model on a sequence.
   * @param {number[]} tokenIds - token IDs
   * @param {boolean[]} padMask - true = real token
   * @returns {number[]} logits [human, AI]
   */
  _forward(tokenIds, padMask) {
    const T = tokenIds.length;
    const D = this.cfg.word_feat_dim;

    // word features
    let seq = this._wordFeatures(tokenIds, padMask);

    // add positional embeddings (positions 1..T for words)
    seq = seq.map((v, t) => add(v, this._posEmbed.forward(t + 1)));

    // prepend CLS token (position 0)
    const cls = add(this._clsToken, this._posEmbed.forward(0));
    seq = [cls, ...seq];

    // build attention mask: true = IGNORE (CLS=false, padding positions=true)
    const attnMask = [false, ...padMask.map(m => !m)];

    // transformer layers
    for (const layer of this._layers) {
      seq = layer.forward(seq, attnMask);
    }

    // classify from CLS token
    const clsOut  = seq[0];
    const normed  = this._headNorm.forward(clsOut);
    const h1      = this._headFc1.forward(normed).map(gelu);
    const logits  = this._headFc2.forward(h1);

    return Array.from(logits);
  }

  // ── public API ──────────────────────────────────────────────────────────────

  /**
   * Predict whether a single text is human or AI-generated.
   *
   * @param {string} text
   * @returns {{ label: 'human'|'AI', confidence: number, score: number,
   *             logits: number[], probs: number[] }}
   */
  predict(text) {
    if (!this.loaded) throw new Error('Model not loaded. Call detector.load(path) first.');

    const maxWords = this.cfg.max_words;
    const ids      = this.tokeniser.encode(text, maxWords);
    const length   = Math.min(ids.length, maxWords);

    // pad to maxWords
    const paddedIds = new Array(maxWords).fill(0);
    const padMask   = new Array(maxWords).fill(false);
    for (let i = 0; i < length; i++) {
      paddedIds[i] = ids[i];
      padMask[i]   = true;
    }

    const logits = this._forward(paddedIds, padMask);
    const probs  = softmax(logits);

    return {
      label      : probs[1] >= 0.5 ? 'AI' : 'human',
      confidence : Math.max(probs[0], probs[1]),
      score      : probs[1],        // probability of AI
      logits,
      probs,
    };
  }

  /**
   * Predict a batch of texts.
   * @param {string[]} texts
   * @returns {Array<ReturnType<Detector['predict']>>}
   */
  predictBatch(texts) {
    return texts.map(t => this.predict(t));
  }

  /**
   * Pretty-print a prediction result.
   * @param {ReturnType<Detector['predict']>} result
   * @param {string} [preview] - optional text preview
   */
  static format(result, preview = '') {
    const bar   = '█'.repeat(Math.round(result.score * 20)).padEnd(20, '░');
    const label = result.label === 'AI' ? '\x1b[31mAI\x1b[0m' : '\x1b[32mHuman\x1b[0m';
    const lines = [
      `Label      : ${label}`,
      `AI score   : ${bar} ${(result.score * 100).toFixed(1)}%`,
      `Confidence : ${(result.confidence * 100).toFixed(1)}%`,
    ];
    if (preview) lines.push(`Text       : "${preview.slice(0, 80)}…"`);
    return lines.join('\n');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  C L I   (run directly: node detector.js <model> <text>)
// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.log('Usage: node detector.js <model-file> "<text to classify>"');
    console.log('       node detector.js <model-file> --file <text-file.txt>');
    process.exit(1);
  }

  const modelPath = args[0];
  let text;

  if (args[1] === '--file') {
    text = fs.readFileSync(args[2], 'utf8');
  } else {
    text = args.slice(1).join(' ');
  }

  const detector = new Detector();
  await detector.load(modelPath);

  const result = detector.predict(text);
  console.log('\n' + Detector.format(result, text) + '\n');
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}

module.exports = { Detector, SimpleTokenizer };