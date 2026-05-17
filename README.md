# Máquina de Busca - Indexador e Processador de Consultas

**Trabalho Prático 2 - Recuperação da Informação** **Autora:** Déborah Brito Yamamoto (UFMG)  

Este repositório contém a implementação dos módulos centrais de um mecanismo de busca na web: um **Indexador** otimizado para lidar com grandes volumes de dados sob restrição de memória, e um **Processador de Consultas** eficiente utilizando ranqueamento probabilístico e vetorial.

---

## Tecnologias e Técnicas Utilizadas
- **Linguagem:** Python 3.14
- **Bibliotecas Principais:** `nltk` (Processamento de Linguagem Natural), `psutil` (Gerenciamento de Memória), `heapq` (Filas de Prioridade) e `concurrent.futures` (Paralelismo).
- **Gerenciamento de Memória:** Implementação de *External Merge Sort* aliado a avaliação preguiçosa (*lazy evaluation* via geradores) para construir índices maiores que a RAM disponível.
- **Processamento de Consultas:** Algoritmo *Document-at-a-Time* (DAAT) com correspondência conjuntiva (AND lógico).
- **Ranqueamento:** Modelos TF-IDF e BM25 (com parâmetros $k_1 = 1.2$ e $b = 0.75$).

---

## Configuração do Ambiente

1. Certifique-se de estar utilizando o Python 3.14 ou superior.
2. Crie e ative um ambiente virtual:
   ```bash
   python3 -m venv pa2
   source pa2/bin/activate
