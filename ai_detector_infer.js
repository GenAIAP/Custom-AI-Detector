#!/usr/bin/env node

/**
 * AI Detector — Inference (JavaScript Port)
 * =========================================
 * Performs all Transformer and tensor operations from scratch.
 * 
 * Usage:
 *    node ai_detector_infer.js "Your text to classify here"
 *    node ai_detector_infer.js --file text.txt
 */

const fs = require('fs');
const readline = require('readline');

// ══════════════════════════════════════════════════════════════════════════════
// 1. TENSOR MATH FROM SCRATCH (Zero-dependency Forward Pass)
// ══════════════════════════════════════════════════════════════════════════════

// Exact PyTorch GELU uses the Error Function (erf)
function erf(x) {
    const sign = Math.sign(x);
    x = Math.abs(x);
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
}

function gelu_arr(arr) {
    let out = new Float32Array(arr.length);
    const sqrt2 = Math.SQRT2;
    for (let i = 0; i < arr.length; i++) {
        let x = arr[i];
        out[i] = 0.5 * x * (1 + erf(x / sqrt2));
    }
    return out;
}

function softmax_arr(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    if (max === -Infinity) return new Float32Array(arr.length);

    let out = new Float32Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        let val = Math.exp(arr[i] - max);
        out[i] = val;
        sum += val;
    }
    for (let i = 0; i < arr.length; i++) out[i] /= sum;
    return out;
}

function linear(x, W, b, R, in_dim, out_dim) {
    let out = new Float32Array(R * out_dim);
    for (let r = 0; r < R; r++) {
        for (let oc = 0; oc < out_dim; oc++) {
            let sum = b ? b[oc] : 0;
            for (let ic = 0; ic < in_dim; ic++) {
                sum += x[r * in_dim + ic] * W[oc * in_dim + ic];
            }
            out[r * out_dim + oc] = sum;
        }
    }
    return out;
}

function linear_1d(x, W, b, in_dim, out_dim) {
    let out = new Float32Array(out_dim);
    for (let oc = 0; oc < out_dim; oc++) {
        let sum = b ? b[oc] : 0;
        for (let ic = 0; ic < in_dim; ic++) {
            sum += x[ic] * W[oc * in_dim + ic];
        }
        out[oc] = sum;
    }
    return out;
}

function layer_norm(arr, weight, bias, R, C, eps = 1e-5) {
    let out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) {
        let mean = 0;
        for (let c = 0; c < C; c++) mean += arr[r * C + c];
        mean /= C;

        let var_sum = 0;
        for (let c = 0; c < C; c++) {
            let diff = arr[r * C + c] - mean;
            var_sum += diff * diff;
        }
        // PyTorch uses biased variance for LayerNorm
        let std = Math.sqrt(var_sum / C + eps);

        for (let c = 0; c < C; c++) {
            out[r * C + c] = ((arr[r * C + c] - mean) / std) * weight[c] + bias[c];
        }
    }
    return out;
}

function normalize_rows(arr, R, C, eps = 1e-12) {
    let out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) {
        let sum = 0;
        for (let c = 0; c < C; c++) {
            let val = arr[r * C + c];
            sum += val * val;
        }
        let norm = Math.sqrt(Math.max(sum, eps));
        for (let c = 0; c < C; c++) {
            out[r * C + c] = arr[r * C + c] / norm;
        }
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. TOKENIZER
// ══════════════════════════════════════════════════════════════════════════════

class SimpleTokenizer {
    constructor(vocabData) {
        this.vocab_size = vocabData.vocab_size;
        this.token2id = vocabData.token2id;
        this.PAD = 0; this.UNK = 1; this.BOS = 2; this.EOS = 3;
    }

    preTokenize(text) {
        const regex = /(?:[\p{L}\p{N}_]+)|[^\p{L}\p{N}_\s]/gu;
        return text.toLowerCase().match(regex) || [];
    }

    encode(text, max_len = 256) {
        let tokens = [this.BOS];
        for (let w of this.preTokenize(text)) {
            tokens.push(this.token2id[w] !== undefined ? this.token2id[w] : this.UNK);
        }
        tokens.push(this.EOS);
        tokens = tokens.slice(0, max_len);
        while (tokens.length < max_len) {
            tokens.push(this.PAD);
        }
        return tokens;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. MODEL ARCHITECTURE (Forward Pass)
// ══════════════════════════════════════════════════════════════════════════════

function model_forward(token_ids, weights, cfg) {
    const T = cfg.max_seq_len;
    const D = cfg.embed_dim;
    const T_dim = cfg.traj_dim;

    // ── 3.1 Embedding ──
    let emb = new Float32Array(T * D);
    let embed_w = weights["embed.weight"].data;
    for (let i = 0; i < T; i++) {
        let id = token_ids[i];
        for (let d = 0; d < D; d++) emb[i * D + d] = embed_w[id * D + d];
    }

    // ── 3.2 Trajectory Extractor ──
    const T_eff = T - 1;
    let traj_cat = new Float32Array(T_eff * (D + 1));
    for (let i = 0; i < T_eff; i++) {
        let distSq = 0;
        for (let d = 0; d < D; d++) {
            let delta = emb[(i + 1) * D + d] - emb[i * D + d];
            traj_cat[i * (D + 1) + d] = delta;
            distSq += delta * delta;
        }
        traj_cat[i * (D + 1) + D] = Math.sqrt(distSq);
    }

    let proj1 = linear(traj_cat, weights["traj_extractor.proj.0.weight"].data, weights["traj_extractor.proj.0.bias"].data, T_eff, D + 1, T_dim * 2);
    let proj1_act = gelu_arr(proj1);
    let proj2 = linear(proj1_act, weights["traj_extractor.proj.2.weight"].data, weights["traj_extractor.proj.2.bias"].data, T_eff, T_dim * 2, T_dim);
    let traj = layer_norm(proj2, weights["traj_extractor.norm.weight"].data, weights["traj_extractor.norm.bias"].data, T_eff, T_dim);

    // ── 3.3 Pattern Codebook ──
    const n_codes = cfg.pattern_codes;
    const top_k = cfg.pattern_top_k;
    let cb_w = weights["codebook.codebook"].data;
    let cb_norm = normalize_rows(cb_w, n_codes, T_dim);
    let traj_norm = normalize_rows(traj, T_eff, T_dim);

    let assigned = new Float32Array(T_eff * T_dim);
    for (let i = 0; i < T_eff; i++) {
        let row_sims = new Float32Array(n_codes);
        for (let j = 0; j < n_codes; j++) {
            let dot = 0;
            for (let d = 0; d < T_dim; d++) {
                dot += traj_norm[i * T_dim + d] * cb_norm[j * T_dim + d];
            }
            row_sims[j] = dot;
        }

        let indexed = Array.from(row_sims).map((val, idx) => ({ val, idx }));
        indexed.sort((a, b) => b.val - a.val);
        let topk = indexed.slice(0, top_k);
        let softmax_w = softmax_arr(topk.map(x => x.val));

        for (let k = 0; k < top_k; k++) {
            let w = softmax_w[k];
            let code_idx = topk[k].idx;
            for (let d = 0; d < T_dim; d++) {
                assigned[i * T_dim + d] += w * cb_w[code_idx * T_dim + d];
            }
        }
    }

    // ── 3.4 Pre-Transformer Addition & Positional Embeddings ──
    let x = new Float32Array(T_eff * T_dim);
    let pos_w = weights["pos_embed.weight"].data;
    for (let i = 0; i < T_eff; i++) {
        for (let d = 0; d < T_dim; d++) {
            x[i * T_dim + d] = traj[i * T_dim + d] + assigned[i * T_dim + d] + pos_w[i * T_dim + d];
        }
    }

    let pad_mask = new Uint8Array(T_eff);
    for (let i = 0; i < T_eff; i++) {
        pad_mask[i] = token_ids[i] === 0 ? 1 : 0;
    }

    // ── 3.5 Transformer Encoder Blocks ──
    for (let l = 0; l < cfg.n_layers; l++) {
        let in_proj_w = weights[`encoder.${l}.attn.in_proj_weight`].data;
        let in_proj_b = weights[`encoder.${l}.attn.in_proj_bias`].data;
        let out_proj_w = weights[`encoder.${l}.attn.out_proj.weight`].data;
        let out_proj_b = weights[`encoder.${l}.attn.out_proj.bias`].data;

        // Multihead Attention
        let QKV = linear(x, in_proj_w, in_proj_b, T_eff, T_dim, 3 * T_dim);
        let mha_inter = new Float32Array(T_eff * T_dim);
        const H = cfg.n_heads;
        const d_k = T_dim / H;

        for (let h = 0; h < H; h++) {
            for (let i = 0; i < T_eff; i++) {
                let scores = new Float32Array(T_eff);
                for (let j = 0; j < T_eff; j++) {
                    let dot = 0;
                    for (let d = 0; d < d_k; d++) {
                        let q = QKV[i * (3 * T_dim) + 0 * T_dim + h * d_k + d];
                        let k_val = QKV[j * (3 * T_dim) + 1 * T_dim + h * d_k + d];
                        dot += q * k_val;
                    }
                    scores[j] = pad_mask[j] ? -Infinity : dot / Math.sqrt(d_k);
                }

                let attn_w = softmax_arr(scores);

                for (let d = 0; d < d_k; d++) {
                    let v_sum = 0;
                    for (let j = 0; j < T_eff; j++) {
                        v_sum += attn_w[j] * QKV[j * (3 * T_dim) + 2 * T_dim + h * d_k + d];
                    }
                    mha_inter[i * T_dim + h * d_k + d] = v_sum;
                }
            }
        }

        let mha_out = linear(mha_inter, out_proj_w, out_proj_b, T_eff, T_dim, T_dim);

        // Add & LayerNorm 1
        for (let i = 0; i < x.length; i++) x[i] += mha_out[i];
        x = layer_norm(x, weights[`encoder.${l}.norm1.weight`].data, weights[`encoder.${l}.norm1.bias`].data, T_eff, T_dim);

        // FFN
        let ff1 = linear(x, weights[`encoder.${l}.ff.0.weight`].data, weights[`encoder.${l}.ff.0.bias`].data, T_eff, T_dim, cfg.ff_dim);
        let ff1_act = gelu_arr(ff1);
        let ff2 = linear(ff1_act, weights[`encoder.${l}.ff.3.weight`].data, weights[`encoder.${l}.ff.3.bias`].data, T_eff, cfg.ff_dim, T_dim);

        // Add & LayerNorm 2
        for (let i = 0; i < x.length; i++) x[i] += ff2[i];
        x = layer_norm(x, weights[`encoder.${l}.norm2.weight`].data, weights[`encoder.${l}.norm2.bias`].data, T_eff, T_dim);
    }

    // ── 3.6 Output Normalization & Pooling ──
    x = layer_norm(x, weights["out_norm.weight"].data, weights["out_norm.bias"].data, T_eff, T_dim);

    let pooled = new Float32Array(T_dim);
    let valid_count = 0;
    for (let i = 0; i < T_eff; i++) {
        if (!pad_mask[i]) {
            valid_count++;
            for (let d = 0; d < T_dim; d++) pooled[d] += x[i * T_dim + d];
        }
    }
    if (valid_count > 0) {
        for (let d = 0; d < T_dim; d++) pooled[d] /= valid_count;
    }

    // ── 3.7 Classifier Head ──
    let c1 = linear_1d(pooled, weights["classifier.0.weight"].data, weights["classifier.0.bias"].data, T_dim, Math.floor(T_dim / 2));
    let c1_act = gelu_arr(c1);
    let logits = linear_1d(c1_act, weights["classifier.3.weight"].data, weights["classifier.3.bias"].data, Math.floor(T_dim / 2), 2);

    return softmax_arr(logits);
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. PIPELINE / INFERENCE LOGIC
// ══════════════════════════════════════════════════════════════════════════════

function loadTrainedModel(filePath) {
    console.log(`Loading model from ${filePath}…`);
    const payload = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

    const cfg = payload.config;
    const tokenizer = new SimpleTokenizer(payload.tokenizer);
    
    // Convert arrays to Float32Array locally for memory optimization
    for (let key in payload.weights) {
        payload.weights[key].data = new Float32Array(payload.weights[key].data);
    }

    let threshold = 0.5;
    if (payload.metrics && payload.metrics.calibrated_threshold !== undefined) {
        threshold = payload.metrics.calibrated_threshold;
    } else if (cfg.decision_threshold !== undefined) {
        threshold = cfg.decision_threshold;
    }

    const balancedAcc = payload.metrics && payload.metrics.final_balanced_accuracy !== undefined 
                        ? payload.metrics.final_balanced_accuracy.toFixed(6) 
                        : 'undefined';
    console.log(`Model loaded | balanced_acc=${balancedAcc} | decision threshold=${threshold.toFixed(2)}`);
    
    return { weights: payload.weights, tokenizer, cfg, threshold };
}

function predict(text, weights, tokenizer, cfg, threshold) {
    const ids = tokenizer.encode(text, cfg.max_seq_len);
    const probs = model_forward(ids, weights, cfg);
    const prob_ai = probs[1];
    const prob_human = probs[0];
    
    const pred = prob_ai >= threshold ? 1 : 0;
    
    return {
        prediction: pred === 1 ? "AI" : "Human",
        confidence: Number(((pred === 1 ? prob_ai : prob_human) * 100).toFixed(2)),
        prob_human: Number((prob_human * 100).toFixed(2)),
        prob_ai: Number((prob_ai * 100).toFixed(2)),
        threshold: threshold
    };
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. CLI HANDLER
// ══════════════════════════════════════════════════════════════════════════════

function getSentences(text) {
    return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 0);
}

async function main() {
    let textInput = "";
    let fileInput = "";
    
    // Hardcode the default exactly to what the Python file expects
    let modelPath = "trained_model (13).json"; 
    let customThreshold = null;

    const args = process.argv.slice(2);
    for (let i = 0; i < args.length; i++) {
        if (args[i] === "--file" || args[i] === "-f") {
            fileInput = args[++i];
        } else if (args[i] === "--model" || args[i] === "-m") {
            modelPath = args[++i];
        } else if (args[i] === "--threshold" || args[i] === "-t") {
            customThreshold = parseFloat(args[++i]);
        } else if (args[i] === "--") {
            continue; 
        } else if (!args[i].startsWith("-")) {
            textInput = args[i];
        }
    }

    // Smart-check: If the user typed a filename as text, auto-correct it.
    if (textInput && !fileInput && textInput.endsWith('.txt') && fs.existsSync(textInput)) {
        fileInput = textInput;
        textInput = "";
    }

    // STRICT FILE CHECK: No more silent fallbacks!
    if (!fs.existsSync(modelPath)) {
        console.error(`\n❌ ERROR: Could not find the model file: "${modelPath}"`);
        console.error(`Make sure the file is in the same folder, or specify it manually with:`);
        console.error(`node ai_detector_infer.js --model "your_model_name.json" --file text.txt\n`);
        process.exit(1);
    }

    const { weights, tokenizer, cfg, threshold: calThreshold } = loadTrainedModel(modelPath);
    const threshold = customThreshold !== null ? customThreshold : calThreshold;

    let textsToProcess = [];

    if (fileInput) {
        const fileContent = fs.readFileSync(fileInput, 'utf-8');
        textsToProcess = getSentences(fileContent);
    } else if (textInput) {
        textsToProcess = getSentences(textInput);
    }

    const processTexts = (texts) => {
        console.log(`\n${'='.repeat(60)}  (threshold=${threshold.toFixed(2)})`);
        let all_ai_probs = [];

        for (let text of texts) {
            const result = predict(text, weights, tokenizer, cfg, threshold);
            all_ai_probs.push(result.prob_ai);
            
            const truncText = text.length > 80 ? text.substring(0, 80) + '…' : text;
            console.log(`\nText    : ${truncText}`);
            console.log(`Result  : ${result.prediction}  (${result.confidence}% confident)`);
            console.log(`Probs   : Human=${result.prob_human}%  AI=${result.prob_ai}%`);
        }

        if (all_ai_probs.length > 0) {
            const avg_ai = all_ai_probs.reduce((a, b) => a + b, 0) / all_ai_probs.length;
            const overall_pred = (avg_ai / 100.0) >= threshold ? "AI" : "Human";
            console.log(`\n${'-'.repeat(60)}`);
            console.log(`OVERALL ANALYSIS (${all_ai_probs.length} sentences)`);
            console.log(`Average AI Probability: ${avg_ai.toFixed(2)}%`);
            console.log(`Final Conclusion: ${overall_pred}`);
        }
        console.log("=".repeat(60));
    };

    if (textsToProcess.length > 0) {
        processTexts(textsToProcess);
    } else {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        console.log("\nEnter text to classify (empty line to quit):");
        
        const promptUser = () => {
            rl.question("> ", (line) => {
                line = line.trim();
                if (!line) {
                    rl.close();
                    return;
                }
                processTexts(getSentences(line));
                promptUser();
            });
        };
        promptUser();
    }
}

main().catch(err => {
    console.error("An error occurred:", err);
});