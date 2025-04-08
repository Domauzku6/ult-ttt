import tkinter as tk
from tkinter import messagebox
import random
import copy
import math
import time
import os
import logging # Added for logging functionality
import datetime # Added for timestamps in filenames and logs
import re       # Added for parsing filenames with regular expressions

# --- Constants ---
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ''
BOARD_SIZE = 3  # 3x3 small boards
CELL_SIZE = 3   # 3x3 cells within each small board
DEPTH_LIMIT = 4 # Adjusted default depth slightly (can be tuned)

# Difficulty levels and their chance (percentage) of making a random move
DIFFICULTY_LEVELS = {
    "Easy": 40,
    "Normal": 20,
    "Hard": 0
}
DEFAULT_DIFFICULTY = "Normal"

# Create a dummy logger for simulation calls within minimax where logging isn't needed
dummy_logger = logging.getLogger('dummy')
dummy_logger.addHandler(logging.NullHandler())
dummy_logger.propagate = False # Prevent dummy logs from reaching root logger


# --- Game Logic Class ---
class UltimateTicTacToe:
    def __init__(self):
        self.reset_game()
        self.game_over_logged = False

    def reset_game(self):
        """Initializes or resets the game state."""
        self.boards = [[[[EMPTY for _ in range(CELL_SIZE)] for _ in range(CELL_SIZE)]
                        for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.main_board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = PLAYER_X
        self.active_board_coords = None
        self.game_over = False
        self.winner = None
        self.game_over_logged = False # Reset log flag

    def is_valid_move(self, big_row, big_col, small_row, small_col):
        """Checks if a move is valid according to UTTT rules."""
        if self.game_over: return False
        # Check bounds (although GUI should prevent this)
        if not (0 <= big_row < BOARD_SIZE and 0 <= big_col < BOARD_SIZE and \
                0 <= small_row < CELL_SIZE and 0 <= small_col < CELL_SIZE):
            return False
        if self.boards[big_row][big_col][small_row][small_col] != EMPTY: return False
        if self.active_board_coords:
            active_br, active_bc = self.active_board_coords
            if big_row != active_br or big_col != active_bc: return False
        if self.main_board[big_row][big_col] != EMPTY:
             if self.active_board_coords is None: return False
        return True

    def make_move(self, big_row, big_col, small_row, small_col, logger): # Pass logger instance
        """
        Makes a move, updates board state, checks for wins, sets the next active board,
        and logs the game over state if it occurs.
        Returns True if the move was successful, False otherwise.
        """
        if not self.is_valid_move(big_row, big_col, small_row, small_col):
            logger.error(f"Internal Error: Attempted invalid move ({big_row},{big_col},{small_row},{small_col})")
            return False

        self.boards[big_row][big_col][small_row][small_col] = self.current_player

        local_winner = self.check_local_winner(big_row, big_col)
        game_ended_this_move = False
        if local_winner and self.main_board[big_row][big_col] == EMPTY:
            self.main_board[big_row][big_col] = local_winner
            global_winner = self.check_global_winner()
            if global_winner:
                self.game_over = True; self.winner = global_winner; self.active_board_coords = None; game_ended_this_move = True
            elif self.check_global_draw():
                self.game_over = True; self.winner = "Draw"; self.active_board_coords = None; game_ended_this_move = True
        elif self.main_board[big_row][big_col] == EMPTY and self.check_local_draw(big_row, big_col):
            self.main_board[big_row][big_col] = "Draw"
            if self.check_global_draw():
                self.game_over = True; self.winner = "Draw"; self.active_board_coords = None; game_ended_this_move = True

        # Log Game Over State
        if game_ended_this_move and not self.game_over_logged:
            if self.winner == "Draw": logger.info("--- Game Over: It's a Draw! ---")
            else: logger.info(f"--- Game Over: {self.winner} wins! ---")
            self.game_over_logged = True

        # Determine the next active board & Switch player
        if not self.game_over:
            next_active_big_row, next_active_big_col = small_row, small_col
            if self.main_board[next_active_big_row][next_active_big_col] != EMPTY: self.active_board_coords = None
            else: self.active_board_coords = (next_active_big_row, next_active_big_col)
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        return True

    def check_winner_on_lines(self, board, size):
        lines = []; lines.extend(board); lines.extend([[board[i][j] for i in range(size)] for j in range(size)])
        lines.append([board[i][i] for i in range(size)]); lines.append([board[i][size - 1 - i] for i in range(size)])
        for line in lines:
            if len(set(line)) == 1 and line[0] != EMPTY and line[0] != "Draw": return line[0]
        return None

    def check_local_winner(self, big_row, big_col):
        if self.main_board[big_row][big_col] != EMPTY: return self.main_board[big_row][big_col] if self.main_board[big_row][big_col] != "Draw" else None
        return self.check_winner_on_lines(self.boards[big_row][big_col], CELL_SIZE)

    def check_local_draw(self, big_row, big_col):
        if self.main_board[big_row][big_col] == PLAYER_X or self.main_board[big_row][big_col] == PLAYER_O: return False
        if self.main_board[big_row][big_col] == "Draw": return True
        if self.check_winner_on_lines(self.boards[big_row][big_col], CELL_SIZE): return False
        for r in range(CELL_SIZE):
            for c in range(CELL_SIZE):
                if self.boards[big_row][big_col][r][c] == EMPTY: return False
        return True

    def check_global_winner(self): return self.check_winner_on_lines(self.main_board, BOARD_SIZE)
    def check_global_draw(self):
        if self.check_global_winner(): return False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.main_board[r][c] == EMPTY: return False
        return True

    def get_possible_moves(self):
        moves = []
        if self.game_over: return moves
        original_active_coords = self.active_board_coords
        if self.active_board_coords:
            br, bc = self.active_board_coords
            if self.main_board[br][bc] == EMPTY:
                for sr in range(CELL_SIZE):
                    for sc in range(CELL_SIZE):
                        if self.boards[br][bc][sr][sc] == EMPTY: moves.append((br, bc, sr, sc))
            if not moves or self.main_board[br][bc] != EMPTY: self.active_board_coords = None
        if self.active_board_coords is None:
            current_moves = []
            for br_any in range(BOARD_SIZE):
                for bc_any in range(BOARD_SIZE):
                    if self.main_board[br_any][bc_any] == EMPTY:
                        for sr_any in range(CELL_SIZE):
                            for sc_any in range(CELL_SIZE):
                                if self.boards[br_any][bc_any][sr_any][sc_any] == EMPTY: current_moves.append((br_any, bc_any, sr_any, sc_any))
            self.active_board_coords = original_active_coords; return current_moves
        self.active_board_coords = original_active_coords; return moves

    def evaluate_board_state(self):
        global_winner = self.check_global_winner()
        if global_winner == PLAYER_O: return 10000
        if global_winner == PLAYER_X: return -10000
        if self.check_global_draw(): return 0
        score = 0; main_board_potential = self._evaluate_lines(self.main_board, BOARD_SIZE, 100); score += main_board_potential
        local_board_score = 0
        for br in range(BOARD_SIZE):
            for bc in range(BOARD_SIZE):
                 if self.main_board[br][bc] == EMPTY: local_potential = self._evaluate_lines(self.boards[br][bc], CELL_SIZE, 1); local_board_score += local_potential
        score += local_board_score * 0.1; return score

    def _evaluate_lines(self, board, size, base_score):
        score = 0; lines = []; lines.extend(board); lines.extend([[board[i][j] for i in range(size)] for j in range(size)])
        lines.append([board[i][i] for i in range(size)]); lines.append([board[i][size - 1 - i] for i in range(size)])
        for line in lines:
            o_count = line.count(PLAYER_O); x_count = line.count(PLAYER_X); empty_count = line.count(EMPTY)
            if o_count == size: score += base_score * 10; continue
            elif x_count == size: score -= base_score * 10; continue
            if o_count == size - 1 and empty_count == 1: score += base_score * 5
            elif x_count == size - 1 and empty_count == 1: score -= base_score * 5
            elif o_count == size - 2 and empty_count == 2: score += base_score
            elif x_count == size - 2 and empty_count == 2: score -= base_score
        return score

    def minimax(self, depth, is_maximizing, alpha, beta, current_game_state):
        g_winner = current_game_state.check_global_winner()
        if g_winner == PLAYER_O: return 10000 + depth
        if g_winner == PLAYER_X: return -10000 - depth
        if current_game_state.check_global_draw(): return 0
        if depth == 0: return current_game_state.evaluate_board_state()
        possible_moves = current_game_state.get_possible_moves()
        if not possible_moves:
            g_winner_check = current_game_state.check_global_winner()
            if g_winner_check == PLAYER_O: return 10000 + depth;
            if g_winner_check == PLAYER_X: return -10000 - depth;
            if current_game_state.check_global_draw(): return 0;
            return 0
        if is_maximizing:
            max_eval = -math.inf
            for move in possible_moves:
                br, bc, sr, sc = move; next_state = copy.deepcopy(current_game_state)
                move_made = next_state.make_move(br, bc, sr, sc, dummy_logger)
                if not move_made: continue
                eval_score = self.minimax(depth - 1, False, alpha, beta, next_state)
                max_eval = max(max_eval, eval_score); alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = math.inf
            for move in possible_moves:
                br, bc, sr, sc = move; next_state = copy.deepcopy(current_game_state)
                move_made = next_state.make_move(br, bc, sr, sc, dummy_logger)
                if not move_made: continue
                eval_score = self.minimax(depth - 1, True, alpha, beta, next_state)
                min_eval = min(min_eval, eval_score); beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    # Added logger parameter for logging fallback
    def find_best_move(self, logger):
        """Finds the best move for the AI (Player O) using Minimax."""
        best_score = -math.inf; best_move = None; possible_moves = self.get_possible_moves()
        if not possible_moves: return None
        random.shuffle(possible_moves); alpha = -math.inf; beta = math.inf
        for move in possible_moves:
            br, bc, sr, sc = move; next_state = copy.deepcopy(self)
            move_made = next_state.make_move(br, bc, sr, sc, dummy_logger)
            if not move_made: continue
            eval_score = self.minimax(DEPTH_LIMIT - 1, False, alpha, beta, next_state)
            if eval_score > best_score: best_score = eval_score; best_move = move
            alpha = max(alpha, eval_score)
        if best_move is None:
            # Log the fallback event
            logger.warning("Minimax failed to select a best move; resorting to random valid move.")
            best_move = possible_moves[0] # Already shuffled
        return best_move


# --- GUI Class ---
class UltimateTicTacToeGUI:
    def __init__(self, root, log_dir, start_game_number):
        self.root = root
        self.root.title("Ultimate Tic Tac Toe")
        self.log_dir = log_dir
        self.game_number = start_game_number - 1
        self.logger = None
        self.game = UltimateTicTacToe()

        # --- Difficulty Setting ---
        self.difficulty_var = tk.StringVar(value=DEFAULT_DIFFICULTY)
        self.difficulty_chances = DIFFICULTY_LEVELS

        # Color definitions
        self.active_bg = "#e0e0ff"; self.inactive_bg = "#ffffff"; self.won_x_bg = "#ffdddd"
        self.won_o_bg = "#ddffdd"; self.draw_bg = "#cccccc"

        self.buttons = {}; self.main_board_frames = {}

        self.create_widgets()
        self.reset_game_gui() # Initial setup for the first game
        self.update_gui()

    def setup_logger(self, game_num, log_directory):
        """Configures a unique logger for the current game."""
        logger_name = f'UTTT_Game_{game_num}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        for handler in list(self.logger.handlers): self.logger.removeHandler(handler); handler.close()
        now = datetime.datetime.now(); timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"game{game_num}.{timestamp}.txt"
        log_filepath = os.path.join(log_directory, log_filename)
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # Added levelname
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        print(f"Logging Game {game_num} to: {log_filepath}")

    def create_widgets(self):
        """Creates the GUI elements (board, status label, reset button, difficulty)."""
        # --- Main Game Board Frame ---
        self.main_frame = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        self.main_frame.pack(pady=5, padx=10) # Reduced vertical padding

        for big_r in range(BOARD_SIZE):
            for big_c in range(BOARD_SIZE):
                board_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SOLID, borderwidth=1, bg=self.inactive_bg)
                board_frame.grid(row=big_r, column=big_c, padx=2, pady=2)
                self.main_board_frames[(big_r, big_c)] = board_frame
                for small_r in range(CELL_SIZE):
                    for small_c in range(CELL_SIZE):
                        coords = (big_r, big_c, small_r, small_c)
                        button = tk.Button(
                            board_frame, text=EMPTY, width=3, height=1, font=('Arial', 16, 'bold'),
                            command=lambda c=coords: self.on_button_click(c)
                        )
                        button.grid(row=small_r, column=small_c)
                        self.buttons[coords] = button

        # --- Controls Frame (Status, Reset, Difficulty) ---
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=5, padx=10, fill=tk.X)

        # Status Label (takes available width)
        self.status_label = tk.Label(self.controls_frame, text="", font=('Arial', 14))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Reset Button (fixed size)
        self.reset_button = tk.Button(self.controls_frame, text="New Game", command=self.reset_game_gui)
        self.reset_button.pack(side=tk.RIGHT, padx=5)

        # --- Difficulty Frame ---
        self.difficulty_frame = tk.Frame(self.root)
        self.difficulty_frame.pack(pady=5, padx=10)

        tk.Label(self.difficulty_frame, text="AI Difficulty:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        for level in self.difficulty_chances.keys():
            rb = tk.Radiobutton(
                self.difficulty_frame, text=level, variable=self.difficulty_var,
                value=level, font=('Arial', 11),
                command=self.on_difficulty_change # Reset game when difficulty changes
            )
            rb.pack(side=tk.LEFT, padx=5)


    def on_difficulty_change(self):
        """Called when a difficulty radio button is selected."""
        print(f"Difficulty changed to: {self.difficulty_var.get()}. Starting new game.")
        self.logger.info(f"--- Difficulty changed to {self.difficulty_var.get()} ---")
        # Reset the game to apply the new difficulty immediately
        self.reset_game_gui()

    def on_button_click(self, coords):
        """Handles player (X) clicking on a cell."""
        if self.game.current_player == PLAYER_X and not self.game.game_over:
            big_r, big_c, small_r, small_c = coords
            if self.game.is_valid_move(big_r, big_c, small_r, small_c):
                move_successful = self.game.make_move(big_r, big_c, small_r, small_c, self.logger)
                if move_successful:
                    absolute_row = big_r * CELL_SIZE + small_r; absolute_col = big_c * CELL_SIZE + small_c
                    self.logger.info(f"{PLAYER_X} moved to (Row: {absolute_row + 1}, Col: {absolute_col + 1})")
                    self.update_gui()
                    if not self.game.game_over and self.game.current_player == PLAYER_O:
                        self.root.after(100, self.ai_turn)

    def ai_turn(self):
        """Handles the AI's (O) turn, considering difficulty."""
        if self.game.current_player == PLAYER_O and not self.game.game_over:
            self.status_label.config(text="AI's Turn (O) - Thinking...")
            self.root.update_idletasks()

            start_time = time.time()
            move = None
            possible_moves = self.game.get_possible_moves() # Get possible moves first

            if not possible_moves:
                 self.logger.warning("AI found no possible moves.")
                 self.update_gui() # Update status (might be game over)
                 return # Exit AI turn if no moves are possible

            # --- Difficulty Check ---
            selected_difficulty = self.difficulty_var.get()
            random_chance = self.difficulty_chances.get(selected_difficulty, 0) # Default to 0 if key not found
            make_random_move = random.randint(1, 100) <= random_chance

            if make_random_move:
                move = random.choice(possible_moves)
                self.logger.info(f"AI ({selected_difficulty}) making random move due to difficulty setting.")
            else:
                # Find the best move using minimax (pass logger for fallback logging)
                move = self.game.find_best_move(self.logger)

            end_time = time.time()
            self.logger.info(f"AI calculation time: {end_time - start_time:.3f} seconds")

            # --- Execute Move ---
            if move:
                br, bc, sr, sc = move
                if self.game.is_valid_move(br, bc, sr, sc):
                    absolute_row = br * CELL_SIZE + sr; absolute_col = bc * CELL_SIZE + sc
                    # Log the chosen move type (random or minimax)
                    move_type = "random (difficulty)" if make_random_move else "minimax/fallback"
                    self.logger.info(f"{PLAYER_O} ({move_type}) moved to (Row: {absolute_row + 1}, Col: {absolute_col + 1})")
                    self.game.make_move(br, bc, sr, sc, self.logger)
                    self.update_gui()
                else:
                     self.logger.error(f"AI chose an invalid move: {move}. Active: {self.game.active_board_coords}, Board State: {self.game.main_board[br][bc]}")
                     self.update_gui()
            else:
                # This case should be less likely now as we check possible_moves earlier
                self.logger.error("AI failed to determine a move even after checks.")
                self.update_gui()

    def update_gui(self):
        """Updates the entire GUI based on the current game state."""
        for coords, button in self.buttons.items():
            br, bc, sr, sc = coords; cell_value = self.game.boards[br][bc][sr][sc]
            button.config(text=cell_value)
            is_player_turn = (self.game.current_player == PLAYER_X)
            button.config(state=tk.DISABLED if cell_value != EMPTY or self.game.game_over or not is_player_turn else tk.NORMAL)
            if cell_value == PLAYER_X: button.config(fg='blue')
            elif cell_value == PLAYER_O: button.config(fg='red')
            else: button.config(fg='black')

        active_highlight_applied = False
        for (br, bc), frame in self.main_board_frames.items():
            main_board_status = self.game.main_board[br][bc]; is_active = False
            if not self.game.game_over:
                if self.game.active_board_coords: is_active = (self.game.active_board_coords == (br, bc))
                else: is_active = (main_board_status == EMPTY)
            bg_color = self.inactive_bg
            if main_board_status == PLAYER_X: bg_color = self.won_x_bg
            elif main_board_status == PLAYER_O: bg_color = self.won_o_bg
            elif main_board_status == "Draw": bg_color = self.draw_bg
            elif is_active: bg_color = self.active_bg; active_highlight_applied = True
            frame.config(bg=bg_color)
            for child in frame.winfo_children():
                if isinstance(child, tk.Button):
                    child_coords = next((c for c, b in self.buttons.items() if b == child), None)
                    if child_coords:
                        b_br, b_bc, b_sr, b_sc = child_coords
                        if self.game.boards[b_br][b_bc][b_sr][b_sc] == EMPTY: child.config(bg=bg_color)
                        else: non_active_bg = bg_color if bg_color != self.active_bg else self.inactive_bg; child.config(bg=non_active_bg)

        if self.game.game_over:
            if self.game.winner == "Draw": self.status_label.config(text="Game Over: It's a Draw!")
            else: self.status_label.config(text=f"Game Over: {self.game.winner} wins!")
        else:
            player_turn_text = "Your Turn (X)" if self.game.current_player == PLAYER_X else f"AI's Turn (O) [{self.difficulty_var.get()}]" # Show difficulty
            active_board_text = ""
            if self.game.active_board_coords: active_r, active_c = self.game.active_board_coords; active_board_text = f" - Play in board ({active_r + 1}, {active_c + 1})"
            elif active_highlight_applied: active_board_text = " - Play in any highlighted board"
            self.status_label.config(text=f"{player_turn_text}{active_board_text}")

    def reset_game_gui(self):
        """Resets the game logic, sets up logging for the new game, and updates the GUI."""
        self.game_number += 1
        self.setup_logger(self.game_number, self.log_dir)
        current_difficulty = self.difficulty_var.get() # Get current difficulty
        self.logger.info(f"--- New Game Started (Game {self.game_number}, Difficulty: {current_difficulty}) ---")
        print(f"\n--- New Game Started (Game {self.game_number}, Difficulty: {current_difficulty}) ---")

        self.game.reset_game()
        for coords, button in self.buttons.items(): button.config(text=EMPTY, state=tk.NORMAL, fg='black', bg=self.inactive_bg)
        for frame in self.main_board_frames.values(): frame.config(bg=self.inactive_bg)
        self.update_gui()

# --- Function to find the highest existing game number ---
def find_latest_game_number(log_directory):
    """Scans the log directory and returns the highest game number found in filenames."""
    max_num = 0; log_pattern = re.compile(r'^game(\d+)\..*\.txt$')
    try:
        if os.path.isdir(log_directory):
            for filename in os.listdir(log_directory):
                match = log_pattern.match(filename)
                if match: game_num = int(match.group(1)); max_num = max(max_num, game_num)
    except FileNotFoundError: print(f"Log directory not found: {log_directory}. Starting game count from 1.")
    except Exception as e: print(f"Error scanning log directory: {e}")
    return max_num

# --- Main Execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    latest_game_num = find_latest_game_number(log_dir)
    start_game_num = latest_game_num + 1
    print(f"Found latest game number: {latest_game_num}. Starting new session with game number: {start_game_num}")

    root = tk.Tk()
    gui = UltimateTicTacToeGUI(root, log_dir, start_game_num)
    root.mainloop()
