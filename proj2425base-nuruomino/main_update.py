# main_update.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
import traceback

try:
    # Ensure stdout is using UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("Starting main_update.py")
    sys.stdout.flush()
    
    print("Importing modules...")
    sys.stdout.flush()
    
    from nuruomino_update import Board, NuruominoState, Nuruomino, TETROMINOS
    from search import *
    
    print("Modules imported successfully")
    sys.stdout.flush()
    
    def test_nuruomino_solver():
        print("Loading board from file...")
        sys.stdout.flush()
        
        # Carregar um tabuleiro de teste simples
        test_file = "c:\\Users\\afons\\Documents\\IST\\IA\\proj2425base-02-05\\sample-nuruominoboards\\test-01.txt"
        try:
            board = Board.parse_from_file(test_file)
            
            if board is None:
                print(f"Failed to load board from {test_file}")
                sys.stdout.flush()
                return
        except Exception as e:
            print(f"Error loading board: {e}")
            sys.stdout.flush()
            traceback.print_exc()
            return
        
        print("Board loaded successfully")
        sys.stdout.flush()
        
        # Imprimir o tabuleiro original
        print("Original board:")
        sys.stdout.flush()
        board.print_board()
        print()
        sys.stdout.flush()
        
        # Criar o problema do Nuruomino
        problem = Nuruomino(board)
        
        # Resolver usando busca em profundidade
        print("Solving using depth-first search...")
        sys.stdout.flush()
        
        try:
            solution = depth_first_tree_search(problem)
            
            if solution:
                print("Solution found!")
                sys.stdout.flush()
                
                # Imprimir o tabuleiro resolvido
                final_state = solution.state
                board.print_state_board(final_state)
                
                # Mostrar as peças colocadas em cada região
                placed_pieces = board.get_placed_pieces(final_state)
                print("Pieces placed by region:")
                sys.stdout.flush()
                for region, piece_type in placed_pieces.items():
                    print(f"Region {region}: {piece_type}")
                    sys.stdout.flush()
            else:
                print("No solution found!")
                sys.stdout.flush()
        except Exception as e:
            print(f"Error during solving: {e}")
            sys.stdout.flush()
            traceback.print_exc()
    
    if __name__ == "__main__":
        test_nuruomino_solver()
        
except Exception as e:
    print(f"Error in main script: {e}")
    traceback.print_exc()

if __name__ == "__main__":
    test_nuruomino_solver()
