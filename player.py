import chess
import torch
from chess_tournament import Player
from transformers import AutoModelForCausalLM, AutoTokenizer

# Got this from WikiPedia
PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

class TransformerPlayer(Player):

    def __init__(self, name: str = "Chessy", model_name: str = "CondeSoulrack/chessy-v1"):
        super().__init__(name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.temperature = 1.0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def score_moves(self, fen: str, moves: list[chess.Move]) -> list[float]:
        """
        Score each move by how likely the model thinks it is.
        """
        prompt = f"FEN: {fen} MOVE:"
        prompt_len = len(self.tokenizer(prompt).input_ids)

        texts = [f"FEN: {fen} MOVE: {m.uci()}" for m in moves]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            logits = self.model(**batch).logits
        log_probs = torch.log_softmax(logits / self.temperature, dim=-1)

        pad_id = self.tokenizer.pad_token_id
        scores = []

        for i in range(len(moves)):
            ids = batch["input_ids"][i]
            total = 0.0
            n = 0
            for j in range(prompt_len, len(ids)):
                token_id = ids[j].item()
                if token_id == pad_id:
                    break
                total += log_probs[i, j - 1, token_id].item()
                n += 1
            scores.append(total / n if n > 0 else float("-inf"))

        return scores

    def tactical_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """
        Heuristic bonus for moves that are tactically attractive.
        """
        bonus = 0.0

        # Capturing a queen is better than a pawn
        if board.is_capture(move):
            if board.is_en_passant(move):
                bonus += 0.15
            else:
                victim = board.piece_at(move.to_square)
                if victim:
                    bonus += PIECE_VALUE.get(victim.piece_type, 0) * 0.15

        # Giving check puts more pressure
        board.push(move)
        if board.is_check():
            bonus += 0.3
        board.pop()

        # Promoting a pawn to a queen
        if move.promotion:
            bonus += PIECE_VALUE.get(move.promotion, 0) * 0.15

        return bonus

    def get_move(self, fen: str) -> str | None:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Checkmate in one
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        model_scores = self.score_moves(fen, legal_moves)

        # Combine model training with tactical bonuses
        scored = [
            (move, model_score + self.tactical_bonus(board, move))
            for move, model_score in zip(legal_moves, model_scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Lookahead on top 5 best moves from the other player
        top_candidates = scored[:5]
        best_move = top_candidates[0][0]
        best_value = float("-inf")

        for move, our_score in top_candidates:
            board.push(move)

            # Skip moves that end the game in a draw
            if board.is_game_over():
                board.pop()
                continue

            # Score the options from the resulting position
            opp_moves = list(board.legal_moves)
            if opp_moves:
                opp_scores = self.score_moves(board.fen(), opp_moves)
                opp_best = max(opp_scores)
                value = our_score - opp_best
            else:
                # Opponent has no moves after ours
                value = our_score + 10

            board.pop()

            if value > best_value:
                best_value = value
                best_move = move

        return best_move.uci()
