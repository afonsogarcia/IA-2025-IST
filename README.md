# 🧩 Projeto NURUOMINO - Inteligência Artificial 2024/25
**Nota- 17.3**

José Afonso Garcia

Tomás Antunes

## 📖 Descrição do Projeto

O projeto NURUOMINO é uma implementação de um resolvedor automático de puzzles do tipo **LITS** (Logic puzzles Involving Tetrominoes Shapes), um jogo lógico de colocação de peças inspirado em tetraminós. O objetivo é desenvolver um programa que, utilizando técnicas de **procura informada** (algoritmo A*), encontre uma disposição válida de tetraminós em um tabuleiro NxN dividido em regiões poligonais.

## 🎯 Objetivo

Dado um tabuleiro dividido em regiões numeradas (cada uma com pelo menos 4 células), o programa deve encontrar uma solução que coloque exatamente um tetraminó (L, I, T ou S) em cada região, respeitando todas as regras do puzzle.

## 📋 Regras do Puzzle

1. **Uma peça por região**: Cada região deve conter exatamente um tetraminó
2. **Não adjacência**: Tetraminós iguais (considerando rotações e reflexões) não podem estar ortogonalmente adjacentes
3. **Conectividade**: Todas as células preenchidas devem formar uma única área ortogonalmente conectada
4. **Regra 2x2**: É proibido que qualquer bloco 2×2 de células esteja completamente preenchido

## 🔧 Estrutura do Código

### Classes Principais

- **`NuruominoState`**: Representa um estado do puzzle com identificador único
- **`Board`**: Representação otimizada do tabuleiro com cache para regiões e adjacências
- **`Nuruomino`**: Classe principal que implementa o problema de procura

### Algoritmos e Otimizações

- **Algoritmo A*** com heurística admissível baseada no número de regiões não preenchidas
- **MRV (Most Restrictive Variable)**: Prioriza regiões com menos opções de colocação
- **Cache de variantes**: Pré-computação de todas as rotações e reflexões dos tetraminós
- **Verificações eficientes**: Validação de regras durante a colocação das peças

### Tetraminós Suportados

- **L**: Forma em L com 4 células
- **I**: Linha reta com 4 células  
- **T**: Forma em T com 4 células
- **S**: Forma em S com 4 células

## 🚀 Como Executar

### Pré-requisitos
- Python 3.7 ou superior
- Windows PowerShell (para os comandos listados)

### Executar um Teste Específico
```powershell
# Navegar para o diretório do projeto
cd proj2425base-nuruomino

# Executar um teste específico
Get-Content "..\sample-nuruominoboards\test-01.txt" | python nuruomino.py

# Ou para testes do diretório public
Get-Content "..\public\test04.txt" | python nuruomino.py
```

### Executar Todos os Testes
```powershell
# Navegar para o diretório do projeto
cd proj2425base-nuruomino

# Executar o script de testes
python .\run_tests.py
```

### Entrada Manual
```powershell
# Para inserir manualmente um tabuleiro
python nuruomino.py
# Digite o tabuleiro linha por linha, separando os valores por tabs
```

## 📁 Estrutura de Arquivos

```
proj2425base-nuruomino/
├── nuruomino.py          # Código principal do resolvedor
├── run_tests.py          # Script para executar todos os testes
├── search.py             # Algoritmos de procura (A*, greedy)
├── utils.py              # Utilitários e orientações
└── __pycache__/          # Arquivos compilados Python

sample-nuruominoboards/   # Testes de exemplo
├── test-01.txt           # Arquivo de entrada
├── test-01.out.txt       # Saída esperada
└── ...

public/                   # Testes públicos
├── test04.txt            # Arquivo de entrada
├── test04.out            # Saída esperada
└── ...
```

## 🔍 Formato de Entrada e Saída

### Entrada
- Cada linha representa uma linha do tabuleiro
- Valores separados por tabs representam o identificador da região
- Exemplo:
```
1	1	2	2
1	2	2	3
1	3	3	3
```

### Saída
- Mesmo formato da entrada, mas com as letras dos tetraminós (L, I, T, S) nas posições preenchidas
- Posições não preenchidas mantêm o número da região original

## 🧠 Detalhes Técnicos

### Heurística
- **Admissível**: Baseia-se no número de regiões não preenchidas
- **Penalização**: Adiciona pequena penalização para regiões com poucas opções
- **Infinito**: Retorna valor infinito para estados inválidos

### Otimizações Implementadas
- Cache de regiões e adjacências
- Pré-computação de variantes de tetraminós
- Estratégia MRV para seleção de variáveis
- Verificações eficientes de conectividade e regras

## 📝 Notas de Desenvolvimento

- O código inclui prints opcionais na função `result()` para visualizar o progresso da solução
- Implementação otimizada baseada em boas práticas de eficiência
- Suporte para diferentes formatos de arquivos de teste (.out, .out.txt)

## 🎮 Exemplo de Execução

```powershell
PS> Get-Content "..\sample-nuruominoboards\test-01.txt" | python nuruomino.py

--- Colocando tetraminó 'L' na região 1 ---
Posições: [(0, 0), (0, 1), (1, 0), (2, 0)]
Estado atual do tabuleiro:
L	L	2	2	3	3
L	2	2	2	3	3
L	3	3	2	3	5
...
```

---
*Projeto desenvolvido para a disciplina de Inteligência Artificial, Instituto Superior Técnico, 2024/25*
