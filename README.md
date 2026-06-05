I will continue to update this repo if I can think of more ways to detect AI.
This is entirely vibe coded, but all the ideas and mechanics are mine.\
Still trying to figure out why the false positive rates are sooo high though AI texts generally get higher AI scores.

The current system uses a trajectory pattern detection system and it basically just find the direction and distance from a word to another word in the embedding space. Like up 1.0-> down 1.2 -> left 3.4 etc. The distance is calculated based on basic subtraction and I'm considering changing it to the similarity score like dot product in attention mechanism. I chose this mechanism instead of burstiness or perplexity like other AI detectors because this is more resistant to synonym swapping or sentence rephrasing as the trajectories still move in somewhat that direction after swapping with a synonym 
