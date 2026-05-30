const fs = require('fs');
const onnx = require('onnxruntime-node');

// 1. Load the trained patterns (KMeans Centers)
const centersData = JSON.parse(fs.readFileSync('trained_model.json', 'utf8'));
const vCenters = centersData.v_clusters_centers;
const aCenters = centersData.a_clusters_centers;

// 2. Load Word Embeddings (GloVe 50d) manually into memory
console.log("Loading GloVe Word Embeddings. This might take 1-2 seconds...");
const glove = {};
const gloveFile = fs.readFileSync('glove.6B.50d.txt', 'utf8');
gloveFile.split('\n').forEach(line => {
    const parts = line.trim().split(' ');
    if (parts.length === 51) {
        const word = parts[0];
        const vec = parts.slice(1).map(Number);
        glove[word] = vec;
    }
});
console.log("✅ Embeddings Loaded!");

// Helper: Standard Euclidean distance mapping for clustering
function getMinCluster(vec, centers) {
    let minD = Infinity, minIdx = 0;
    for (let i = 0; i < centers.length; i++) {
        let sum = 0;
        for (let j = 0; j < 50; j++) sum += (vec[j] - centers[i][j]) ** 2;
        if (sum < minD) { minD = sum; minIdx = i; }
    }
    return minIdx;
}

// 3. Main Text Analysis Logic
async function detectAI(text) {
    // Simple regex Tokenization (acts similarly to NLTK here)
    const tokens = text.toLowerCase().match(/\b\w+\b/g) || [];
    const vecs = tokens.filter(w => glove[w]).map(w => glove[w]);

    let v_seq = [0], a_seq = [0], m_seq = [0.0]; 

    if (vecs.length >= 3) {
        const V = [], A = [];
        const M = []; // Magnitude tracking
        
        // Calculate Component 1: Directional Velocity (V) and Magnitude (M)
        for (let i = 0; i < vecs.length - 1; i++) {
            const diff = [];
            let sumSq = 0; // For Magnitude
            for (let j = 0; j < 50; j++) {
                const shift = vecs[i+1][j] - vecs[i][j];
                diff.push(shift);
                sumSq += shift * shift;
            }
            V.push(diff);
            M.push(Math.sqrt(sumSq)); // The "Forced Synonym" Tracker
        }
        
        // Calculate Component 2: Pattern of Patterns Acceleration (A)
        for (let i = 0; i < V.length - 1; i++) {
            const diff = [];
            for (let j = 0; j < 50; j++) diff.push(V[i+1][j] - V[i][j]);
            A.push(diff);
        }

        // Map sequences to their respective cluster models
        v_seq = V.map(v => getMinCluster(v, vCenters));
        a_seq = A.map(a => getMinCluster(a, aCenters));
        m_seq = M;
    }

    // Pad 'A' array by 1 to match the length of V and M mathematically
    while (a_seq.length < v_seq.length) a_seq.push(0);

    // 4. Run Neural Network Inference natively using ONNX
    const session = await onnx.InferenceSession.create('./model.onnx');
    
    // Format them correctly for PyTorch matching types
    const v_tensor = new onnx.Tensor('int64', BigInt64Array.from(v_seq.map(BigInt)), [1, v_seq.length]);
    const a_tensor = new onnx.Tensor('int64', BigInt64Array.from(a_seq.map(BigInt)), [1, a_seq.length]);
    const m_tensor = new onnx.Tensor('float32', Float32Array.from(m_seq), [1, m_seq.length]);

    const feeds = { v_seq: v_tensor, a_seq: a_tensor, m_seq: m_tensor };
    const results = await session.run(feeds);
    
    // PyTorch outputs raw logits. Convert it to Percentage using a manual Sigmoid function:
    const logit = Number(results.output.data[0]);
    const probability = 1 / (1 + Math.exp(-logit));
    
    console.log(`\nText Analyzed: "${text.substring(0, 50)}..."`);
    console.log(`🤖 AI Confidence Score: ${(probability * 100).toFixed(2)}%`);
}

// === RUN TEST ===
detectAI("As an AI language model, I am designed to assist with a variety of tasks efficiently and correctly.");