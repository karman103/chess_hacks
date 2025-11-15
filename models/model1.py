# src/engine.py
from __future__ import annotations
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# ---- AlphaZero move encoding (4672) ----
DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
KNIGHT_DELTAS = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
PROM_PIECES = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}

def square_to_xy(sq:int): return (sq//8, sq%8)
def xy_to_square(r,c): return r*8+c
def flip_square_for_black(sq:int)->int:
    r,c=square_to_xy(sq); return xy_to_square(7-r,7-c)

def move_to_policy_index(board: chess.Board, move: chess.Move):
    fr, to = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        fr = flip_square_for_black(fr)
        to = flip_square_for_black(to)
    fr_r, fr_c = square_to_xy(fr)
    tr, tc = square_to_xy(to)
    base = (fr_r*8+fr_c)*73
    dr, dc = tr-fr_r, tc-fr_c

    # queen-like (56)
    for d_idx, (rr,cc) in enumerate(DIRS):
        if rr==0 and cc==0: continue
        if rr==0 and dr==0 and dc!=0 and ((cc==1 and dc>0) or (cc==-1 and dc<0)):
            dist=abs(dc); 
            if 1<=dist<=7: return base + d_idx*7 + (dist-1)
        elif cc==0 and dc==0 and dr!=0 and ((rr==1 and dr>0) or (rr==-1 and dr<0)):
            dist=abs(dr); 
            if 1<=dist<=7: return base + d_idx*7 + (dist-1)
        elif rr!=0 and cc!=0 and abs(dr)==abs(dc) and (dr//rr == dc//cc):
            dist=abs(dr)
            if 1<=dist<=7: return base + d_idx*7 + (dist-1)

    # knight (8)
    for k_idx,(rr,cc) in enumerate(KNIGHT_DELTAS):
        if dr==rr and dc==cc:
            return base + 56 + k_idx

    # underpromotions (9)
    if move.promotion in PROM_PIECES:
        pp = PROM_PIECES[move.promotion]
        dir_plane=None
        if dr==1 and dc==0:   dir_plane=0   # forward
        elif dr==1 and dc==1: dir_plane=1   # diag-right
        elif dr==1 and dc==-1:dir_plane=2   # diag-left
        if dir_plane is not None:
            return base + 56 + 8 + dir_plane*3 + pp

    return None

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((18,8,8), dtype=np.float32)
    b=board; mirror = b.turn==chess.BLACK
    pieces=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
    for i,pc in enumerate(pieces):
        own=b.pieces(pc, b.turn); opp=b.pieces(pc, not b.turn)
        for sq in own:
            s=flip_square_for_black(sq) if mirror else sq
            r,c=square_to_xy(s); planes[i,r,c]=1.0
        for sq in opp:
            s=flip_square_for_black(sq) if mirror else sq
            r,c=square_to_xy(s); planes[6+i,r,c]=1.0
    planes[12,:,:]=1.0
    planes[13,:,:]=1.0 if b.has_kingside_castling_rights(b.turn) else 0.0
    planes[14,:,:]=1.0 if b.has_queenside_castling_rights(b.turn) else 0.0
    planes[15,:,:]=1.0 if b.has_kingside_castling_rights(not b.turn) else 0.0
    planes[16,:,:]=1.0 if b.has_queenside_castling_rights(not b.turn) else 0.0
    planes[17,:,:]=float(b.fullmove_number % 2)
    return planes

# ---- Tiny network (matches your trained weights) ----
class ResBlock(nn.Module):
    def __init__(self, ch:int):
        super().__init__()
        self.c1=nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.g1=nn.GroupNorm(8,ch)
        self.c2=nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.g2=nn.GroupNorm(8,ch)
    def forward(self,x):
        h=F.silu(self.g1(self.c1(x)))
        h=self.g2(self.c2(h))
        return F.silu(h+x)

class TinyAZ(nn.Module):
    def __init__(self, width=64, blocks=8):
        super().__init__()
        self.stem=nn.Conv2d(18,width,3,padding=1,bias=False)
        self.g0=nn.GroupNorm(8,width)
        self.body=nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.p1=nn.Conv2d(width,32,1,bias=False); self.pg=nn.GroupNorm(8,32)
        self.pf=nn.Linear(32*8*8,4672)
        self.v1=nn.Conv2d(width,32,1,bias=False); self.vg=nn.GroupNorm(8,32)
        self.vf1=nn.Linear(32*8*8,128); self.vf2=nn.Linear(128,1)
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    def forward(self,x):
        h=F.silu(self.g0(self.stem(x)))
        h=self.body(h)
        p=F.silu(self.pg(self.p1(h))).view(x.size(0),-1)
        logits=self.pf(p)
        v=F.silu(self.vg(self.v1(h))).view(x.size(0),-1)
        v=F.silu(self.vf1(v)); v=torch.tanh(self.vf2(v)).squeeze(-1)
        return logits, v

# ---- MCTS ----
class Node:
    __slots__=("prior","visit","value_sum","children")
    def __init__(self, prior: float):
        self.prior=float(prior); self.visit=0; self.value_sum=0.0; self.children={}
    def q(self): return 0.0 if self.visit==0 else self.value_sum/self.visit

def _dirichlet_mix(priors, alpha=0.3, eps=0.25):
    noise=np.random.dirichlet([alpha]*len(priors))
    return (1-eps)*priors + eps*noise

class MCTS:
    def __init__(self, model: TinyAZ, device, sims=400, cpuct=1.5):
        self.model=model; self.device=device; self.sims=sims; self.cpuct=cpuct

    def _policy_value(self, board: chess.Board):
        x=torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, v = self.model(x)
        logits=logits[0].cpu()
        mask=np.zeros((4672,),dtype=np.float32)
        for mv in board.legal_moves:
            idx=move_to_policy_index(board,mv)
            if idx is not None: mask[idx]=1.0
        logp = logits - 1e9*torch.tensor(1.0-mask)
        p = torch.softmax(logp, dim=-1).numpy()
        return p, float(v.item()), mask

    def select(self, board: chess.Board) -> chess.Move:
        root = Node(1.0)
        priors, _, _ = self._policy_value(board)
        for mv in board.legal_moves:
            idx=move_to_policy_index(board,mv)
            if idx is not None: root.children[idx]=Node(priors[idx])
        if root.children:
            mixed=_dirichlet_mix(np.array([c.prior for c in root.children.values()]), 0.3, 0.25)
            for (idx,c),p in zip(root.children.items(), mixed): c.prior=float(p)

        for _ in range(self.sims):
            self._simulate(board.copy(), root)

        best_idx, best_vis = None, -1
        legal=list(board.legal_moves)
        for idx,c in root.children.items():
            if c.visit>best_vis: best_vis=c.visit; best_idx=idx
        for mv in legal:
            if move_to_policy_index(board,mv)==best_idx: return mv
        return random.choice(legal)

    def _simulate(self, board: chess.Board, node: Node) -> float:
        if board.is_game_over():
            r=board.result()
            return 1.0 if r=='1-0' else (-1.0 if r=='0-1' else 0.0)
        if not node.children:
            priors, v, _ = self._policy_value(board)
            node.visit += 1; node.value_sum += v
            for mv in board.legal_moves:
                idx=move_to_policy_index(board,mv)
                if idx is not None: node.children[idx]=Node(priors[idx])
            return v
        total = sum(c.visit for c in node.children.values()) + 1e-8
        best_score, best_idx, best_mv = -1e9, None, None
        for mv in board.legal_moves:
            idx=move_to_policy_index(board,mv)
            if idx is None or idx not in node.children: continue
            c=node.children[idx]
            u=self.cpuct * c.prior * math.sqrt(total)/(1+c.visit)
            score=c.q()+u
            if score>best_score:
                best_score, best_idx, best_mv = score, idx, mv
        if best_mv is None:
            best_mv=random.choice(list(board.legal_moves))
        board.push(best_mv)
        v=self._simulate(board, node.children[best_idx])
        node.children[best_idx].visit += 1
        node.children[best_idx].value_sum += v if board.turn==chess.BLACK else -v
        return v

# ---- Public API for main.py ----
class TinyChessEngine:
    def __init__(self, weights_path: str, width=64, blocks=8, sims=400, cpuct=1.5, device_str='cpu'):
        self.device=torch.device(device_str)
        self.model=TinyAZ(width=width, blocks=blocks).to(self.device)
        sd=torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.mcts=MCTS(self.model, self.device, sims=sims, cpuct=cpuct)

    def best_move(self, board: chess.Board) -> chess.Move:
        return self.mcts.select(board)
