import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import random
import sys
import gc
import signal

# =========================================================
# PHASE 7 CONFIGURATION
# =========================================================
MODEL_PATH = "phase5_model.pth"   # <--- USING PHASE 5 MODEL AS TEACHER
TOTAL_GAMES = 25_000_000          # 25 Million
SAVE_INTERVAL = 1_000_000         # Save every 1 Million
GAMES_PER_CYCLE = 360             # Games per worker batch
NUM_CORES = 9                     # 9 Cores
OUTPUT_PREFIX = "phase7_dataset"  # <--- Outputting for Phase 7

# Global variables for workers
worker_model = None
worker_device = None

# =========================================================
# MODEL ARCHITECTURE (Assuming Phase 5 kept this structure)
# =========================================================
class Net_Phase5(nn.Module):
    def __init__(self):
        super(Net_Phase5, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(81, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 81)
        )
    def forward(self, x): return self.layers(x)

# =========================================================
# OPTIMIZED GAME LOGIC
# =========================================================
PLAYER_X_INT, PLAYER_O_INT, EMPTY_INT, DRAW_INT = 1, -1, 0, 2

class UltimateTicTacToe:
    __slots__ = ['boards', 'main_board', 'current_player_int', 'active_board_coords', 'game_over', 'winner']
    def __init__(self): self.reset_game()
    def reset_game(self):
        self.boards = np.full((3,3,3,3), EMPTY_INT, dtype=np.int8)
        self.main_board = np.full((3,3), EMPTY_INT, dtype=np.int8)
        self.current_player_int = PLAYER_X_INT
        self.active_board_coords = None
        self.game_over = False
        self.winner = None

    def get_possible_moves(self):
        if self.game_over: return []
        b_to_c = []
        if self.active_board_coords and self.main_board[self.active_board_coords] == EMPTY_INT:
            b_to_c.append(self.active_board_coords)
        else:
            rows, cols = np.where(self.main_board == EMPTY_INT)
            b_to_c = list(zip(rows, cols))
        moves = []
        for br, bc in b_to_c:
            srs, scs = np.where(self.boards[br, bc] == EMPTY_INT)
            for i in range(len(srs)): moves.append((br, bc, srs[i], scs[i]))
        return moves

    def make_move(self, br, bc, sr, sc):
        self.boards[br, bc, sr, sc] = self.current_player_int
        self._update_game_state(br, bc, sr, sc)
        if not self.game_over:
            self.current_player_int = -self.current_player_int
            self.active_board_coords = (sr, sc) if self.main_board[sr, sc] == EMPTY_INT else None
        return True

    def _update_game_state(self, br, bc, sr, sc):
        if self.main_board[br, bc] == EMPTY_INT:
            if self._check_lines(self.boards[br, bc]): self.main_board[br, bc] = self.current_player_int
            elif np.all(self.boards[br, bc] != EMPTY_INT): self.main_board[br, bc] = DRAW_INT
        if self._check_lines(self.main_board): 
            self.game_over, self.winner = True, self.current_player_int
        elif np.all(self.main_board != EMPTY_INT): 
            self.game_over, self.winner = True, DRAW_INT

    def _check_lines(self, board):
        p = self.current_player_int
        ps = 3 * p
        return np.any(board.sum(axis=1)==ps) or np.any(board.sum(axis=0)==ps) or np.trace(board)==ps or np.trace(np.fliplr(board))==ps

def flatten_board(game):
    state = np.zeros(81, dtype=np.float32)
    flat = game.boards.flatten()
    current = game.current_player_int
    state[flat == current] = 1.0
    state[flat == -current] = -1.0
    return state

def move_to_index(move):
    br, bc, sr, sc = move
    return (br * 27) + (bc * 9) + (sr * 3) + sc

# =========================================================
# WORKER PROCESS
# =========================================================
def init_worker(model_state_dict):
    global worker_model, worker_device
    worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    
    worker_model = Net_Phase5().to(worker_device)
    worker_model.load_state_dict(model_state_dict)
    worker_model.eval()

def worker_process_cycle(seed):
    global worker_model, worker_device
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    
    cycle_s, cycle_p, cycle_v = [], [], []
    t_win  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t_loss = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    t_draw = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    try:
        for _ in range(GAMES_PER_CYCLE):
            game = UltimateTicTacToe()
            history = [] 
            
            while not game.game_over:
                state_flat = flatten_board(game)
                state_t = torch.from_numpy(state_flat).unsqueeze(0).to(worker_device)
                
                with torch.no_grad():
                    logits = worker_model(state_t)[0].cpu().numpy()

                moves = game.get_possible_moves()
                if not moves: break 

                valid_idxs = [move_to_index(m) for m in moves]
                masked = np.full(81, -float('inf'), dtype=np.float32)
                masked[valid_idxs] = logits[valid_idxs]
                probs = np.exp(masked - np.max(masked))
                probs /= np.sum(probs)
                
                chosen = np.random.choice(81, p=probs)
                move = next(m for m in moves if move_to_index(m) == chosen)
                history.append((state_flat, probs, game.current_player_int))
                game.make_move(*move)

            final_winner = game.winner
            for (s, p, plyr) in history:
                cycle_s.append(s); cycle_p.append(p)
                if final_winner == DRAW_INT: cycle_v.append(t_draw)
                elif final_winner == plyr: cycle_v.append(t_win)
                else: cycle_v.append(t_loss)

        return cycle_s, cycle_p, cycle_v
    except Exception: return [], [], []

# =========================================================
# MAIN CONTROLLER
# =========================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print(f"\n🚀 PHASE 7 DATA GENERATION (Using Phase 5 Model) 🚀")
    print(f"Target: {TOTAL_GAMES:,} games")
    print(f"Workers: {NUM_CORES} (GPU Accelerated)")
    
    # Permission Check
    try:
        with open("permission_check.tmp", "w") as f: f.write("ok")
        os.remove("permission_check.tmp")
    except:
        print("❌ ERROR: Check Docker Volume Mount!"); sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Missing: {MODEL_PATH}"); sys.exit(1)
        
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    total_cycles = TOTAL_GAMES // GAMES_PER_CYCLE
    tasks = [random.randint(0, 9999999) + i for i in range(total_cycles)]
    
    global_s, global_p, global_v = [], [], []
    games_in_buffer = 0
    part_index = 1
    
    with mp.Pool(processes=NUM_CORES, initializer=init_worker, initargs=(model_dict,)) as pool:
        pbar_total = tqdm(total=TOTAL_GAMES, unit="games", desc="TOTAL PROGRESS", position=0)
        pbar_buffer = tqdm(total=SAVE_INTERVAL, unit="games", desc="Next Save Buffer", position=1, leave=False)
        
        for res in pool.imap_unordered(worker_process_cycle, tasks):
            s, p, v = res
            if not s: continue
            
            global_s.extend(s); global_p.extend(p); global_v.extend(v)
            games_just_processed = GAMES_PER_CYCLE 
            games_in_buffer += games_just_processed
            
            pbar_total.update(games_just_processed)
            pbar_buffer.update(games_just_processed)
            
            if games_in_buffer >= SAVE_INTERVAL:
                pbar_buffer.close()
                filename = f"{OUTPUT_PREFIX}_part{part_index}.npz"
                tqdm.write(f"\n💾 Saving {filename} ({len(global_s)} positions)...")
                np.savez_compressed(filename, states=np.array(global_s), policies=np.array(global_p), values=np.array(global_v))
                
                global_s, global_p, global_v = [], [], []
                games_in_buffer = 0
                part_index += 1
                gc.collect()
                pbar_buffer = tqdm(total=SAVE_INTERVAL, unit="games", desc="Next Save Buffer", position=1, leave=False)

        pbar_total.close(); pbar_buffer.close()

    if len(global_s) > 0:
        filename = f"{OUTPUT_PREFIX}_part{part_index}_final.npz"
        print(f"💾 Saving Final Chunk to {filename}...")
        np.savez_compressed(filename, states=np.array(global_s), policies=np.array(global_p), values=np.array(global_v))

    print(f"\n🎉 PHASE 7 DATASET COMPLETE 🎉")
