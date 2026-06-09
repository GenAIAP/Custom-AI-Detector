'use strict';

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const fs = require('fs');
const { Detector } = require('./detector.js');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const detector = new Detector();

// This finds the first file matching the pattern exported by the Python trainer
const getModelPath = () => {
  const files = fs.readdirSync(__dirname);
  const modelFile = files.find(f => f.startsWith('detector.json-'));
  return modelFile ? path.join(__dirname, modelFile) : null;
};

async function init() {
  const modelPath = getModelPath();
  if (!modelPath) {
    console.error('Error: No model file found in directory.');
    console.error('Ensure you have a file like detector.json-33.7MB present.');
    process.exit(1);
  }

  await detector.load(modelPath);

  app.use(express.static(path.join(__dirname, 'public')));

  io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);

    socket.on('predict', (text) => {
      try {
        const result = detector.predict(text);
        socket.emit('prediction-result', result);
      } catch (err) {
        socket.emit('error', err.message);
      }
    });
  });

  const PORT = process.env.PORT || 3000;
  server.listen(PORT, () => {
    console.log(`🚀 AI Detector server running at http://localhost:${PORT}`);
  });
}

init().catch(err => console.error('Failed to start server:', err));