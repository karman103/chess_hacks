# AplhaZero implementaion
Reached 2000+ ELo on 4 hours of supervised trainig and then 8 hours of self-play type RL 
Hardware- AMD MI3000 x8 
Features-
1) stockfish distillation to policy network and val network
2) Self play mechanism that traind the policy and value network
3) Eval on stockfish levels
4) Data generation from stockfish with multi core processing 
5) qunatizing the weights of the model to fp16 from fp32
For running any trained bot 