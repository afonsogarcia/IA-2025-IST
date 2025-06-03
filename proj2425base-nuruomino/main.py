#!/usr/bin/env python3
"""
Main para testar o projeto NURUOMINO
Testa as funcionalidades principais da classe Board
"""

from nuruomino import Board, Nuruomino, NuruominoState, ALL_TETROMINO_VARIANTS

def test_exemplo_2():
    """
    Exemplo 2: Colocação de peças e verificação de valores
    """
    # Carregar tabuleiro
    board = Board.parse_from_file("../sample-nuruominoboards/test-01.txt")
    problem = Nuruomino(board)

    # Criar um estado com a configuração inicial:
    initial_state = NuruominoState(board)
    # Mostrar valor na posição (2, 1):
    print(initial_state.get_value(2, 1))
    
    # Find a valid L shape that fits in region 1
    valid_L_shape = None
    for variant in ALL_TETROMINO_VARIANTS['L']:
        placement = problem._can_place_shape_in_region(initial_state, 1, variant, 'L')
        if placement:
            valid_L_shape = variant
            break
    
    # Realizar ação de colocar a peça L na região 1
    result_state = s1 = problem.result(initial_state, (1, 'L', valid_L_shape))
    
    # Alternativa: Colocar peça L diretamente com forma específica
    # result_state = problem.result(initial_state, (1, 'L', [(0, 0), (0, 1), (1, 0), (2, 0)]))
    
    # Mostrar valor na posição (2, 1):
    print(result_state.get_value(2, 1))
    # Mostrar os valores de posições adjacentes
    print(result_state.adjacent_values(2,2))
    
    # Mostrar o board completo com as peças colocadas
    result_state.print_board()

def main():
    """Função principal de teste"""
    try:
        # Exemplo 2: Colocação de peças e verificação de valores
        test_exemplo_2()
        
    except Exception as e:
        print(f"\n Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
