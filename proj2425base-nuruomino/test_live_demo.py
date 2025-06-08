#!/usr/bin/env python3
"""
Script de demonstração do sistema de visualização em tempo real
"""

import sys
from nuruomino import *
from search import astar_search

def parse_instance():
    """Le e processa uma instância de teste do stdin."""
    lines = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            lines.append(line)
    
    # Converter para grid numérico
    grid = []
    for line in lines:
        row = [int(x) for x in line.split()]
        grid.append(row)
    
    return grid

def solve_with_live_visualization(input_file):
    """Resolve o puzzle com visualização em tempo real."""
    print(f"🔍 Carregando puzzle de: {input_file}")
    
    # Ler o arquivo
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Converter para grid
    grid = []
    for line in lines:
        line = line.strip()
        if line:
            row = [int(x) for x in line.split()]
            grid.append(row)
    
    print(f"📊 Grid carregado: {len(grid)}x{len(grid[0])}")
    
    # Criar o tabuleiro inicial
    initial_board = Board(grid)
    
    # Criar o problema com visualização habilitada
    print("🎮 Iniciando solver com visualização em tempo real...")
    problem = Nuruomino(initial_board, enable_visualization=True)
    
    print("🔎 Executando busca A*...")
    # Resolver o problema
    solution = astar_search(problem)
    
    if solution:
        print(f"✅ Solução encontrada em {len(solution.path())} passos!")
        print("🎯 Puzzle resolvido com sucesso!")
        
        # Mostrar estatísticas finais
        if hasattr(problem, 'live_visualizer'):
            problem.live_visualizer.show_final_statistics()
    else:
        print("❌ Nenhuma solução encontrada.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_live_demo.py <arquivo_puzzle>")
        print("Exemplo: python test_live_demo.py ../sample-nuruominoboards/test-01.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    solve_with_live_visualization(input_file)
