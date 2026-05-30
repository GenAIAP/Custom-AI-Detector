//%%writefile code.js
const fs = require('fs');
const csv = require('csv-parser');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const os = require('os');
const tf = require('@tensorflow/tfjs-node-gpu');

// ==========================================
// GPU ACCELERATED VECTOR & CLUSTERING HELPERS
// ==========================================
function cosineSimilarity(v1, v2) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < v1.length; i++) {
        dotProduct += v1[i] * v2[i];
        normA += v1[i] * v1[i];
        normB += v2[i] * v2[i];
    }
    if (normA === 0 || normB === 0) return 0;
    const sim = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.max(0, Math.min(1, sim)); 
}

/**
 * GPU-Accelerated K-Means using TensorFlow.js Ops.
 * This significantly speeds up clustering for the 1000 AI centroids on a T4.
 */
async function tfKMeans(embeddingsTensor, numClusters, maxIterations = 15, seed = 42) {
    const numPoints = embeddingsTensor.shape[0];
    const pNormSq = embeddingsTensor.square().sum(1).expandDims(1);

    const initialCentIndices = tf.tidy(() => tf.randomUniform([numClusters], 0, numPoints, 'int32', seed));
    let centroids = tf.gather(embeddingsTensor, initialCentIndices);
    initialCentIndices.dispose();

    let clusterAssignments = tf.fill([numPoints], -1, 'int32');

    for (let i = 0; i < maxIterations; i++) {
        const newAssignments = tf.tidy(() => {
            const cNormSq = centroids.square().sum(1).expandDims(0);
            const dotProduct = tf.matMul(embeddingsTensor, centroids, false, true);
            const distances = pNormSq.add(cNormSq).sub(dotProduct.mul(2));
            return distances.argMin(1, 'int32');
        });

        const converged = await tf.tidy(() => tf.equal(newAssignments, clusterAssignments).all()).array();
        
        clusterAssignments.dispose();
        clusterAssignments = newAssignments;

        if (converged && i > 0) break;

        const nextCentroids = tf.tidy(() => {
            const oneHot = tf.oneHot(clusterAssignments, numClusters);
            const counts = oneHot.sum(0).expandDims(1); 
            const sums = tf.matMul(oneHot, embeddingsTensor, true, false); 
            const safeCounts = tf.where(tf.greater(counts, 0), counts, tf.onesLike(counts));
            const computedMeans = sums.div(safeCounts);
            return tf.where(tf.greater(counts, 0), computedMeans, centroids);
        });

        centroids.dispose();
        centroids = nextCentroids;
        
        process.stdout.write(`\rK-Means (GPU) Iteration: ${i + 1}/${maxIterations}...`);
    }
    process.stdout.write('\n');

    const finalCentroids = await centroids.array();
    
    clusterAssignments.dispose();
    centroids.dispose();
    pNormSq.dispose();

    return finalCentroids;
}

// ==========================================
// 1. THE PREFIX-SPAN ALGORITHM
// ==========================================
class PrefixSpan {
    constructor(minSupport, minLength = 2, maxLength = 8) {
        this.minSupport = minSupport; 
        this.minLength = minLength;   
        this.maxLength = maxLength;   
        this.patterns = [];
    }

    mine(database) {
        this.patterns = [];
        this._mineRecursive([], database);
        this.patterns.sort((a, b) => b.support - a.support);
        return this.patterns;
    }

    _mineRecursive(prefix, db) {
        if (prefix.length >= this.maxLength) return;
        const itemCounts = new Map();
        
        for (const sequence of db) {
            const uniqueItems = new Set(sequence); 
            for (const item of uniqueItems) {
                itemCounts.set(item, (itemCounts.get(item) || 0) + 1);
            }
        }

        for (const [item, count] of itemCounts.entries()) {
            if (count >= this.minSupport) {
                const newPrefix = [...prefix, item];
                if (newPrefix.length >= this.minLength) {
                    this.patterns.push({ pattern: newPrefix.join(" -> "), support: count });
                }
                const projectedDB = [];
                for (const sequence of db) {
                    const index = sequence.indexOf(item);
                    if (index !== -1 && index < sequence.length - 1) {
                        projectedDB.push(sequence.slice(index + 1));
                    }
                }
                if (projectedDB.length >= this.minSupport) {
                    this._mineRecursive(newPrefix, projectedDB);
                }
            }
        }
    }
}

// ==========================================
// 2. THE FAST AI DETECTOR (GPU OPTIMIZED)
// ==========================================
class FastAIDetector {
    static SIMILARITY_THRESHOLD = 0.9; 
    static WEIGHT_THRESHOLD = 4.0;     

    constructor() {
        this.centroids = [];
        this.numCentroids = 1000;
        this.knownAiPatterns = [];
        this.knownMetaPatterns = []; 
        this.nnModel = null;
        this.extractor = null;
        this.embeddingCache = new Map();
    }

    async initExtractor() {
        if (this.extractor) return;
        let HF;
        try {
            HF = await import('@huggingface/transformers');
        } catch (e) {
            HF = await import('@xenova/transformers');
        }
        this.extractor = await HF.pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
    }

    async prepareEmbeddingsForText(text) {
        if (!text || typeof text !== 'string') return;
        const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim().split(/\s+/).filter(Boolean);
        const uniqueWords = [...new Set(words)];
        const missingWords = uniqueWords.filter(w => !this.embeddingCache.has(w));
        
        if (missingWords.length === 0) return;
        await this.initExtractor();

        const chunkSize = 50; // Increased batch size for Colab
        for (let i = 0; i < missingWords.length; i += chunkSize) {
            const chunk = missingWords.slice(i, i + chunkSize);
            await Promise.all(chunk.map(async (word) => {
                try {
                    const output = await this.extractor(word, { pooling: 'mean', normalize: true });
                    this.embeddingCache.set(word, Array.from(output.data));
                } catch (err) {
                    this.embeddingCache.set(word, this.getDeterministicFallback(word));
                }
            }));
        }
    }

    getDeterministicFallback(word) {
        let hash = 0;
        for (let i = 0; i < word.length; i++) hash = word.charCodeAt(i) + ((hash << 5) - hash);
        const dimensions = 384;
        const vector = [];
        for (let d = 0; d < dimensions; d++) {
            const seed = Math.sin(hash + d) * 10000;
            vector.push(seed - Math.floor(seed) * 2 - 1); 
        }
        const mag = Math.sqrt(vector.reduce((sum, v) => sum + v*v, 0));
        return mag === 0 ? vector : vector.map(v => v / mag);
    }

    textToConceptChains(text) {
        if (!text || typeof text !== 'string' || this.centroids.length === 0) return [];
        const sentences = text.toLowerCase().split(/[.,?!;]+/);
        const sentenceChains = [];
        for (const sentence of sentences) {
            const words = sentence.replace(/[^a-z0-9\s]/g, '').trim().split(/\s+/);
            const chain = [];
            for (const word of words) {
                if (!word) continue;
                const embedding = this.embeddingCache.get(word) || this.getDeterministicFallback(word);
                let bestCentroidIdx = -1;
                let maxSim = -Infinity;
                for (let i = 0; i < this.centroids.length; i++) {
                    const sim = cosineSimilarity(embedding, this.centroids[i]);
                    if (sim > maxSim) { maxSim = sim; bestCentroidIdx = i; }
                }
                if (bestCentroidIdx !== -1) chain.push({ centroid: bestCentroidIdx, score: maxSim });
            }
            if (chain.length > 0) sentenceChains.push(chain);
        }
        return sentenceChains.map((_, i) => sentenceChains.slice(i, i + 1).flat());
    }

    calculateSimilarity(patternArray, sequence) {
        let patternIdx = 0, matchesScore = 0;
        for (let i = 0; i < sequence.length; i++) {
            if (sequence[i] && sequence[i].centroid == patternArray[patternIdx]) {
                matchesScore += sequence[i].score;
                patternIdx++;
            }
            if (patternIdx === patternArray.length) break;
        }
        return matchesScore / patternArray.length;
    }

    _getFeatureVector(text) {
        const targetChains = this.textToConceptChains(text);
        const featureSize = this.knownAiPatterns.length + this.knownMetaPatterns.length;
        const features = new Array(featureSize).fill(0);
        if (targetChains.length === 0) return features;
        for (const chain of targetChains) {
            this.knownAiPatterns.forEach((p, i) => {
                if (this.calculateSimilarity(p.pattern.split(" -> "), chain) >= FastAIDetector.SIMILARITY_THRESHOLD) features[i]++;
            });
        }
        for (let i = 0; i < this.knownAiPatterns.length; i++) features[i] /= targetChains.length;
        const essayMetaChain = targetChains.map(chain => this.knownAiPatterns.findIndex(p => this.calculateSimilarity(p.pattern.split(" -> "), chain) >= FastAIDetector.SIMILARITY_THRESHOLD)).filter(idx => idx !== -1);
        this.knownMetaPatterns.forEach((metaP, i) => {
            if (this.calculateSimilarity(metaP.pattern.split(" -> "), essayMetaChain) > FastAIDetector.SIMILARITY_THRESHOLD) features[this.knownAiPatterns.length + i] = 1;
        });
        return features;
    }

    _initNN(inputSize) {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputSize] }));
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
        model.compile({ optimizer: tf.train.adam(0.005), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        this.nnModel = model;
    }

    async train(aiTextsArray, humanTextsArray = []) {
        console.log("Warm-up: Caching embeddings on GPU/CPU...");
        const allTexts = [...aiTextsArray, ...humanTextsArray];
        for (let i = 0; i < allTexts.length; i++) {
            await this.prepareEmbeddingsForText(allTexts[i]);
            if (i % 100 === 0) process.stdout.write(`\r- Cached ${i}/${allTexts.length} essays`);
        }
        console.log("\nBuilding AI Vocabulary...");
        const aiWords = new Set();
        for (const text of aiTextsArray) {
            const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim().split(/\s+/);
            for (const w of words) if (w.length > 1) aiWords.add(w);
        }
        const wordVectors = Array.from(aiWords).map(w => this.embeddingCache.get(w)).filter(Boolean);

        console.log(`GPU Clustering ${wordVectors.length} vectors into ${this.numCentroids} centroids...`);
        const wordTensor = tf.tensor2d(wordVectors);
        this.centroids = await tfKMeans(wordTensor, this.numCentroids);
        wordTensor.dispose();

        let aiChains = [];
        for (const text of aiTextsArray) aiChains.push(...this.textToConceptChains(text));
        const dbForMining = aiChains.map(chain => chain.map(item => item.centroid));
        const dynamicSupport = Math.max(5, Math.floor(dbForMining.length * 0.0035));
        
        this.prefixSpan = new PrefixSpan(dynamicSupport, 1, 10); 
        let rawAiPatterns = this.prefixSpan.mine(dbForMining);

        if (humanTextsArray.length > 0) {
            let humanChains = [];
            for (const text of humanTextsArray) humanChains.push(...this.textToConceptChains(text));
            this.knownAiPatterns = rawAiPatterns.slice(0, 300).filter(p => {
                const patternArray = p.pattern.split(" -> ");
                let hCount = 0;
                for (const h of humanChains) if (this.calculateSimilarity(patternArray, h) >= 0.9) hCount++;
                return (p.support / aiChains.length) / Math.max(0.0001, hCount / humanChains.length) > 5.0;
            }).slice(0, 150);
        } else {
            this.knownAiPatterns = rawAiPatterns.slice(0, 150);
        }

        console.log("Training Neural Network on GPU...");
        const trainingFeatures = [], trainingLabels = [];
        for (const t of aiTextsArray) { trainingFeatures.push(this._getFeatureVector(t)); trainingLabels.push(1); }
        for (const t of humanTextsArray) { trainingFeatures.push(this._getFeatureVector(t)); trainingLabels.push(0); }

        this._initNN(trainingFeatures[0].length);
        const xs = tf.tensor2d(trainingFeatures), ys = tf.tensor2d(trainingLabels, [trainingLabels.length, 1]);
        await this.nnModel.fit(xs, ys, { epochs: 30, batchSize: 64, shuffle: true, verbose: 0 });
        xs.dispose(); ys.dispose();
        console.log(`✅ Training Complete.`);
    }

    async detect(newText, verbose = true) {
        if (!this.nnModel) return false;
        await this.prepareEmbeddingsForText(newText);
        const features = this._getFeatureVector(newText);
        const prediction = tf.tidy(() => this.nnModel.predict(tf.tensor2d([features])));
        const score = (await prediction.data())[0];
        prediction.dispose();
        const isAi = score > 0.5; 
        if (verbose) console.log(isAi ? `⚠️ AI DETECTED (${Math.floor(score * 100)}%)` : `✅ HUMAN LIKELY (${Math.floor(score * 100)}%)`);
        return isAi;
    }

    async saveModel(filename = 'trained_model.json') {
        const weights = [];
        for (const w of this.nnModel.getWeights()) weights.push({ data: Array.from(await w.data()), shape: w.shape });
        fs.writeFileSync(filename, JSON.stringify({ centroids: this.centroids, knownAiPatterns: this.knownAiPatterns, knownMetaPatterns: this.knownMetaPatterns, nnWeights: weights }, null, 2));
    }
}

// ==========================================
// 3. WORKER THREAD LOGIC
// ==========================================
if (!isMainThread) {
    const { chunk, humanChains, aiChainsCount, totalHumanChains } = workerData;
    const calculateSimilarity = (patternArray, sequence) => {
        let patternIdx = 0, matchesScore = 0;
        for (let i = 0; i < sequence.length; i++) {
            if (sequence[i] && sequence[i].centroid == patternArray[patternIdx]) { matchesScore += sequence[i].score; patternIdx++; }
            if (patternIdx === patternArray.length) break;
        }
        return matchesScore / patternArray.length;
    };
    const filteredChunk = chunk.filter(pObj => {
        const patternArray = pObj.pattern.split(" -> ");
        let hCount = 0;
        for (const h of humanChains) if (calculateSimilarity(patternArray, h) >= 0.9) hCount += 1;
        return (pObj.support / aiChainsCount) / Math.max(0.0001, hCount / totalHumanChains) > 5.0;
    });
    parentPort.postMessage(filteredChunk);
    return; 
}

// ==========================================
// 4. RUN PIPELINE
// ==========================================
if (isMainThread) {
    const AI_ESSAYS = [], HUMAN_ESSAYS = [];
    function shuffleArray(array, seed = 12345) {
        let m = array.length, t, i;
        const random = () => { const x = Math.sin(seed++) * 10000; return x - Math.floor(x); };
        while (m) { i = Math.floor(random() * m--); t = array[m]; array[m] = array[i]; array[i] = t; }
    }
    const loadCSV = (file, callback) => new Promise(resolve => {
        if (!fs.existsSync(file)) return resolve();
        fs.createReadStream(file).pipe(csv()).on('data', callback).on('end', resolve);
    });

    async function runPipeline() {
        console.log("Loading datasets for Colab GPU run...");
        await loadCSV('dataset.csv', (row) => {
            const textKey = Object.keys(row).find(key => key.toLowerCase().includes('text') || key.toLowerCase().includes('essay'));
            const labelKey = Object.keys(row).find(key => key.toLowerCase().includes('generated') || key.toLowerCase().includes('label'));
            if (!textKey || !labelKey) return; 
            if (String(row[labelKey]).trim() === '0') HUMAN_ESSAYS.push(row[textKey]);
            else if (row[textKey]) AI_ESSAYS.push(row[textKey]);
        });

        shuffleArray(AI_ESSAYS); shuffleArray(HUMAN_ESSAYS);
        const splitIdxAI = Math.floor(AI_ESSAYS.length * 0.95), splitIdxHU = Math.floor(HUMAN_ESSAYS.length * 0.95);
        const trainAI = AI_ESSAYS.slice(0, splitIdxAI), testAI = AI_ESSAYS.slice(splitIdxAI);
        const trainHU = HUMAN_ESSAYS.slice(0, splitIdxHU), testHU = HUMAN_ESSAYS.slice(splitIdxHU);

        const detector = new FastAIDetector();
        await detector.train(trainAI, trainHU);
        await detector.saveModel();
        
        console.log("\nEvaluating Performance...");
        let stats = { tp: 0, fn: 0, tn: 0, fp: 0 };
        for (const text of testAI) if (await detector.detect(text, false)) stats.tp++; else stats.fn++;
        for (const text of testHU) if (await detector.detect(text, false)) stats.fp++; else stats.tn++;
        
        const total = testAI.length + testHU.length;
        console.table({ "Accuracy": (total > 0 ? ((stats.tp + stats.tn) / total) * 100 : 0).toFixed(2) + "%", "Correct AI": stats.tp, "Missed AI": stats.fn, "Correct Human": stats.tn, "Human as AI": stats.fp });
    }

    runPipeline();
}