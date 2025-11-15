# nn/stockfish_evaluator.py
import chess
import chess.engine
from pathlib import Path


class StockfishEvaluator:
    """
    Simple wrapper around Stockfish for supervised labeling.
    Not used in the final engine, only in offline data generation.
    """

    def __init__(self, engine_path: str, time_limit: float = 0.1, depth: int | None = None):
        """
        engine_path: path to Stockfish binary
        time_limit:  seconds per position if depth is None
        depth:       fixed search depth if provided (overrides time_limit)
        """
        self.engine_path = str(engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(Path(engine_path))
        self.time_limit = time_limit
        self.depth = depth

    def evaluate_cp(self, board: chess.Board) -> int:
        """
        Return centipawn eval from the perspective of the side to move.
        Positive means advantage for side to move.
        """
        if self.depth is not None:
            limit = chess.engine.Limit(depth=self.depth)
        else:
            limit = chess.engine.Limit(time=self.time_limit)

        info = self.engine.analyse(board, limit)

        # 'pov' ensures we get score from the side to move's point of view
        score = info["score"].pov(board.turn)

        if score.is_mate():
            # Encode mate as large cp value with sign.
            mate_in = score.mate()
            # sign(mate_in): >0 means side to move is mating
            sign = 1 if mate_in is None or mate_in > 0 else -1
            cp = 10000 * sign
        else:
            # centipawns; mate_score sets the cp equivalent of mate
            cp = score.score(mate_score=10000)
        return int(cp)

    def close(self):
        self.engine.quit()


# --- Tiny smoke test / example usage ---
if __name__ == "__main__":
    # TODO: replace this with the real path to your Stockfish binary
    ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"


    sf = StockfishEvaluator(ENGINE_PATH, time_limit=0.05)

    # Test 1: starting position (should be roughly equal, near 0 cp)
    board = chess.Board()
    cp_start = sf.evaluate_cp(board)
    print("Start position cp (side to move = White):", cp_start)

    # Test 2: after 1. e4 (White should usually be slightly better)
    board.push_san("e4")
    cp_after_e4 = sf.evaluate_cp(board)
    print("After 1. e4 cp (side to move = Black):", cp_after_e4)

    sf.close()
