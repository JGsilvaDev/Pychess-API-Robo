from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from sqlalchemy.orm import Session
from stockfish import Stockfish
from database.database import get_db 
from datetime import datetime
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from Model.users import User
from Model.games import Game
from Model.moves import Move
from Model.evaluation import Evaluation

import math
import math
import os
import chess

app = FastAPI(
    title="Pychess",
    description="API com autenticação JWT",
    version="1.0",
    openapi_tags=[{"name": "DB", "description": "Rotas que acessam o banco de dados"}],
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar os domínios permitidos (origens permitidas)
origins = [
    "http://localhost:3000",  # Frontend Next.js em desenvolvimento
    "http://127.0.0.1:3000",  # Outra variação do localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ou ["*"] se for só pra testes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # <- isso é o importante pro Authorization!
)

load_dotenv()

STOCKFISH_PATH = r"C:\Users\joao.silva\OneDrive - Allparts Componentes Ltda\Documentos\GitHub\Pychess-API\stockfish\stockfish-windows-x86-64-avx2.exe"
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

security = HTTPBearer()

# Inicializa o motor Stockfish
stockfish = Stockfish(STOCKFISH_PATH)
stockfish.set_skill_level(10)  # Ajuste o nível de habilidade (0-20)
stockfish.set_depth(15)  # Profundidade de busca

# Variável para armazenar o histórico do jogo
board = chess.Board()

# Lista global para armazenar até 3 últimas partidas
game_history = []

def fen_to_matrix(fen):
    """Converte um FEN em uma matriz 8x8 representando o tabuleiro."""
    rows = fen.split(" ")[0].split("/")  # Pegamos apenas a parte do tabuleiro no FEN
    board_matrix = []

    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(["."] * int(char))  # Espaços vazios
            else:
                board_row.append(char)  # Peça
        board_matrix.append(board_row)

    return board_matrix

@app.post("/set_difficulty/",tags=['GAME'])
def set_difficulty(level: str):
    """Define o nível de dificuldade do Stockfish"""
    
    difficulty_settings = {
        "muito_baixa": {"skill": 1, "depth": 2, "rating": 150},
        "baixa": {"skill": 2, "depth": 4, "rating": 300},
        "media": {"skill": 5, "depth": 8, "rating": 600},
        "dificil": {"skill": 10, "depth": 14, "rating": 1200},
        "extremo": {"skill": 20, "depth": 22, "rating": "MAX"}
    }

    level = level.lower()
    
    if level not in difficulty_settings:
        raise HTTPException(status_code=400, detail="Nível inválido! Escolha entre: muito_baixa, baixa, media, dificil, extremo.")

    settings = difficulty_settings[level]

    stockfish.set_skill_level(settings["skill"])
    stockfish.set_depth(settings["depth"])

    return {
        "message": f"Dificuldade ajustada para '{level}'",
        "skill_level": settings["skill"],
        "depth": settings["depth"],
        "rating": settings["rating"]
    }

@app.post("/start_game/", tags=['GAME'])
def start_game(user_id: int, db: Session = Depends(get_db)):
    """ Inicia um novo jogo de xadrez e registra a posição inicial. """
    
    # Verifica se há algum jogo em andamento
    existing_game = db.query(Game).filter(Game.player_win == 0).first()
    if existing_game:
        raise HTTPException(status_code=400, detail="Já existe um jogo em andamento!")

    # Criar um novo jogo
    new_game = Game(user_id=user_id)
    db.add(new_game)
    db.commit()
    db.refresh(new_game)

    # Iniciar posição no Stockfish
    stockfish.set_position([])  # posição inicial padrão

    # Criar jogada inicial na tabela moves
    initial_move = Move(
        is_player=None,  # Nenhuma jogada ainda
        move="",  # Movimento vazio (início do jogo)
        board_string=stockfish.get_fen_position(),  # FEN da posição inicial
        mv_quality=None,  # Não se aplica ainda
        game_id=new_game.id
    )
    db.add(initial_move)
    db.commit()

    new_eval = Evaluation(
            game_id=new_game.id,
            evaluation=0,
            depth=0,
            win_probability_white=50,
            win_probability_black=50,
        )
    db.add(new_eval)
    db.commit()

    return {
        "message": "Jogo iniciado!",
        "game_id": new_game.id,
        "board": stockfish.get_board_visual()
    }

@app.get("/game_board/", tags=['GAME'])
def get_game_board(db: Session = Depends(get_db)):
    """ Retorna a visualização do tabuleiro baseado no último estado salvo no banco. """

    # Obtém o último jogo ativo e seu último movimento em uma única consulta
    last_game = (
        db.query(Game.id, Move.board_string)
        .join(Move, Move.game_id == Game.id)
        .filter(Game.player_win == 0)
        .order_by(Game.id.desc(), Move.id.desc())
        .first()
    )

    if not last_game:
        raise HTTPException(status_code=404, detail="Nenhum jogo ativo ou jogada encontrada.")

    game_id, fen_string = last_game

    # Validação do FEN antes de enviar para o Stockfish
    if not fen_string or len(fen_string.split()) != 6:
        raise HTTPException(status_code=400, detail="FEN inválido no banco de dados.")

    # Define a posição no Stockfish
    try:
        stockfish.set_fen_position(fen_string)
        board_visual = stockfish.get_board_visual()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar tabuleiro: {str(e)}")

    if not board_visual:
        raise HTTPException(status_code=500, detail="Falha ao gerar visualização do tabuleiro.")

    return {
        "board": board_visual.split("\n"),
        "fen": fen_string
    }

def rating(user_id: int, db: Session = Depends(get_db)):
    """Avalia o jogo completo armazenado em game_moves e atualiza o rating do jogador no banco de dados."""

    global stockfish

    game = db.query(Game).filter(Game.player_win == 0).first()

    if not game:
        raise HTTPException(status_code=400, detail="Nenhum jogo ativo encontrado!")

    # Obtém os movimentos já registrados no banco para este jogo
    game_moves = db.query(Move.move).filter(Move.game_id == game.id).all()
    game_moves = [m.move for m in game_moves]  # Transformando em lista de strings

    # Verifica se há jogadas para avaliar
    if not game_moves:
        raise HTTPException(status_code=400, detail="Nenhuma jogada registrada para avaliação.")

    # Busca o usuário e seu rating atual
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado!")

    base_rating = user.rating  # Rating atual do jogador
    rating = base_rating  # Inicializa o rating com o valor do banco

    stockfish.set_position([])  # Reseta o Stockfish para o início da partida

    for i, move in enumerate(game_moves):
        if not stockfish.is_move_correct(move):
            raise HTTPException(status_code=400, detail=f"Movimento inválido detectado: {move}")

        stockfish.set_position(game_moves[:i + 1])  # Atualiza posição até a jogada atual

        best_move = stockfish.get_best_move()  # Melhor jogada segundo Stockfish
        evaluation_before = stockfish.get_evaluation()  # Avaliação antes do movimento
        stockfish.make_moves_from_current_position([move])  # Aplica o movimento no Stockfish
        evaluation_after = stockfish.get_evaluation()  # Avaliação depois do movimento
        
        eval_diff = evaluation_before["value"] - evaluation_after["value"]

        if best_move == move:
            rating += 50  # Jogada perfeita
        elif eval_diff > 200:
            rating -= 50  # Erro grave (Blunder)
        elif eval_diff > 100:
            rating -= 20  # Jogada imprecisa
        elif eval_diff > 30:
            rating -= 5   # Pequeno erro
        else:
            rating += 5   # Jogada sólida

    # Garante que o rating final não fique negativo
    final_rating = max(0, rating)

    # Calcula a diferença entre o rating final e o atual do jogador
    rating_diff = final_rating - base_rating

    # Atualiza o rating no banco de dados conforme a diferença
    if rating_diff >= 200:
        user.rating += 100
    elif rating_diff >= 100:
        user.rating += 70
    elif rating_diff >= 20:
        user.rating += 50
    elif rating_diff > 0:
        user.rating += 20

    db.commit()

    return {
        "message": "Avaliação concluída!",
        "final_rating": final_rating,
        "rating_updated": user.rating, 
        "moves_analyzed": len(game_moves)
    }

def analyze_move(move: str,  db: Session = Depends(get_db)):
    """ Analisa a jogada, comparando com a melhor possível. """

     # Verifica se existe um jogo ativo
    game = db.query(Game).filter(Game.player_win == 0).first()

    if not game:
        raise HTTPException(status_code=400, detail="Nenhum jogo ativo encontrado!")

    # Obtém os movimentos já registrados no banco para este jogo
    game_moves = db.query(Move.move).filter(Move.game_id == game.id).all()
    game_moves = [m.move for m in game_moves]  # Transformando em lista de strings

    stockfish.set_position(game_moves)

    # Obtém a melhor jogada recomendada pelo Stockfish
    best_move = stockfish.get_best_move()

    if not stockfish.is_move_correct(move):
        raise HTTPException(status_code=400, detail="Movimento inválido!")

    # Avaliação antes da jogada
    eval_before = stockfish.get_evaluation()
    eval_before_score = eval_before["value"] if eval_before["type"] == "cp" else 0

    # Aplica o movimento do usuário
    game_moves.append(move)
    stockfish.set_position(game_moves)

    # Avaliação após a jogada
    eval_after = stockfish.get_evaluation()
    eval_after_score = eval_after["value"] if eval_after["type"] == "cp" else 0

    # Desfaz o movimento do usuário e testa a melhor jogada do Stockfish
    game_moves.pop()
    stockfish.set_position(game_moves)
    game_moves.append(best_move)
    stockfish.set_position(game_moves)

    # Avaliação após a melhor jogada do Stockfish
    eval_best = stockfish.get_evaluation()
    eval_best_score = eval_best["value"] if eval_best["type"] == "cp" else 0

    # Calcula a diferença entre as avaliações
    diff_user = eval_after_score - eval_before_score  # O quanto a jogada do usuário melhorou ou piorou a posição
    diff_best = eval_best_score - eval_before_score  # O quanto a melhor jogada melhoraria a posição
    diff_to_best = diff_user - diff_best  # Diferença entre a jogada do usuário e a melhor jogada

    # Classificação da jogada
    if diff_to_best == 0:
        classification = "Brilhante 💎"
    elif -30 <= diff_to_best < 0:
        classification = "Boa ✅"
    elif -100 <= diff_to_best < -30:
        classification = "Ok 🤷"
    else:
        classification = "Gafe ❌"

    return {
        "move": move,
        "best_move": best_move,
        "evaluation_before": eval_before_score,
        "evaluation_after": eval_after_score,
        "evaluation_best_move": eval_best_score,
        "classification": classification,
        "board": stockfish.get_board_visual()
    }

@app.post("/play_game/", tags=['GAME'])
async def play_game(move: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """ O usuário joga, e o Stockfish responde com a melhor jogada, verificando capturas. """

    # Verifica se há um jogo ativo
    game = db.query(Game).filter(Game.player_win == 0).first()
    if not game:
        raise HTTPException(status_code=400, detail="Nenhum jogo ativo encontrado!")

    # Obtém o último estado salvo do tabuleiro
    last_move = db.query(Move).filter(Move.game_id == game.id).order_by(Move.id.desc()).first()

    # Se houver um estado salvo, carregamos ele; caso contrário, criamos um novo tabuleiro
    board = chess.Board(last_move.board_string) if last_move and last_move.board_string else chess.Board()

    # Verifica se a jogada do jogador é válida
    if move not in [m.uci() for m in board.legal_moves]:
        raise HTTPException(status_code=400, detail="Movimento do jogador inválido!")

    # Aplica o movimento do jogador no tabuleiro
    board.push(chess.Move.from_uci(move))

    # Atualiza o Stockfish com o novo estado do jogo
    stockfish.set_fen_position(board.fen())

    # Análise da jogada
    analysis = analyze_move(move, db)
    classification = analysis["classification"]

    # Salva o movimento do jogador no banco
    new_move = Move(
        is_player=True,
        move=move,
        board_string=board.fen(),
        mv_quality=classification,
        game_id=game.id,
        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    db.add(new_move)
    db.commit()

    # Verifica xeque-mate após o movimento do jogador
    if board.is_checkmate():
        game.player_win = 1  # Brancas venceram
        db.commit()
        rating(game.user_id)

        return {
            "message": "Xeque-mate! Brancas venceram!",
            "board_fen": board.fen(),
            "player_move": move,
            "stockfish_move": None
        }

    # Stockfish responde com o melhor movimento
    best_move = stockfish.get_best_move()
    if best_move:
        stockfish_move = chess.Move.from_uci(best_move)

        # Se for válido, aplicamos no tabuleiro
        if stockfish_move in board.legal_moves:
            board.push(stockfish_move)
            stockfish.set_fen_position(board.fen())

            # Salva a jogada do Stockfish
            sf_move = Move(
                is_player=False,
                move=best_move,
                board_string=board.fen(),
                game_id=game.id,
                mv_quality=None,
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            db.add(sf_move)
            db.commit()

            # Verifica xeque-mate após a jogada do Stockfish
            if board.is_checkmate():
                game.player_win = 2  # Pretas venceram
                db.commit()
                rating(game.user_id)

                return {
                    "message": "Xeque-mate! Pretas venceram!",
                    "board_fen": board.fen(),
                    "player_move": move,
                    "stockfish_move": best_move
                }
        else:
            return {
                "message": "Movimento do Stockfish inválido. Tentando novamente...",
                "board_fen": board.fen(),
                "player_move": move,
                "stockfish_move": None
            }
        
    background_tasks.add_task(calculate_and_save_evaluation, game.id, db)

    return {
        "message": "Movimentos realizados!",
        "board_fen": board.fen(),
        "player_move": move,
        "stockfish_move": best_move
    }

def calculate_and_save_evaluation(game_id: int, db: Session):
    moves = db.query(Move.move).filter(Move.game_id == game_id).order_by(Move.id).all()
    move_list = [m.move for m in moves]
    stockfish.set_position(move_list)

    best_eval = None
    best_depth = 0

    for depth in range(8, 13):
        stockfish.set_depth(depth)
        evaluation = stockfish.get_evaluation()

        if best_eval is None or abs(evaluation["value"]) > abs(best_eval["value"]):
            best_eval = evaluation
            best_depth = depth

    if best_eval["type"] == "mate":
        if best_eval["value"] > 0:
            win_white = 100
        else:
            win_white = 0
    else:
        cp = best_eval["value"]
        win_white = round((1 / (1 + math.exp(-0.004 * cp))) * 100, 2)

    win_black = round(100 - win_white, 2)

    existing = db.query(Evaluation).filter(Evaluation.game_id == game_id).first()
    if existing:
        existing.evaluation = best_eval["value"]
        existing.depth = best_depth
        existing.win_probability_white = win_white
        existing.win_probability_black = win_black
        existing.last_updated = datetime.utcnow()
    else:
        new_eval = Evaluation(
            game_id=game_id,
            evaluation=best_eval["value"],
            depth=best_depth,
            win_probability_white=win_white,
            win_probability_black=win_black,
        )
        db.add(new_eval)

    db.commit()

# ROTAS A SEREM USADAS AO PENSAR EM INTEGRAR COM O ROBO
@app.get("/get_position/{square}", tags=['ROBOT'])
def get_move_vector(move: str):
    """
    Recebe uma jogada como 'h2h3' e retorna o deslocamento em X, Y
    e o ângulo inteiro que o robô deve girar a partir da posição 0 (referência horizontal).
    """

    if len(move) != 4:
        raise HTTPException(status_code=400, detail="Jogada inválida! Use formato padrão, ex: 'h2h3'.")

    from_square = move[:2]
    to_square = move[2:]

    def get_position(square: str):
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            raise HTTPException(status_code=400, detail=f"Posição inválida: {square}")

        column_map = {
            "a": 1000, "b": 2000, "c": 3000, "d": 4000,
            "e": 5000, "f": 6000, "g": 7000, "h": 8000
        }

        row_map = {
            "1": 1000, "2": 2000, "3": 3000, "4": 4000,
            "5": 5000, "6": 6000, "7": 7000, "8": 8000
        }

        x = column_map[square[0]]
        y = row_map[square[1]]
        return (x, y)

    # Posições de origem e destino
    x1, y1 = get_position(from_square)
    x2, y2 = get_position(to_square)

    # Vetor de deslocamento
    dx = x2 - x1
    dy = y2 - y1

    # Ângulo absoluto (em relação ao eixo X positivo) — referência 0°
    angle_rad = math.atan2(dy, dx)
    angle_deg = int(round(math.degrees(angle_rad)))

    # Corrige ângulos negativos para o intervalo 0°–359°
    if angle_deg < 0:
        angle_deg += 360

    return {
        "from": from_square,
        "to": to_square,
        "dx": dx,
        "dy": dy,
        "angle_deg": angle_deg  # usado sempre a partir da posição 0
    }