# PA2 - Recuperação da Informação

Trabalho da disciplina de Recuperação da Informação. Implementação de um mecanismo de busca na web dividido em dois módulos: um **indexador** (`indexer.py`), que constrói um índice invertido a partir de um corpus de documentos, e um **processador de consultas** (`processor.py`), que usa esse índice para responder consultas e retornar os documentos mais relevantes.

## Visão geral

- **Entrada do indexador**: corpus em formato JSONL, onde cada linha é um documento com os campos `id`, `title` e `text`.
- **Saída do indexador**: índice invertido, índice de documentos e léxico de termos, gravados em disco, respeitando um orçamento de memória configurável.
- **Entrada do processador**: índice gerado pelo indexador + arquivo de consultas (uma consulta por linha).
- **Saída do processador**: para cada consulta, um JSON com os 10 documentos mais relevantes e seus scores, calculados por TFIDF ou BM25.

## Estrutura do projeto

```
.
├── indexer.py           # Construção do índice invertido
├── processor.py         # Processamento de consultas e ranqueamento
├── pa2.pdf               # Enunciado do trabalho
└── Link do Drive.txt     # Link para o corpus utilizado
```

## Requisitos

- Python 3
- Bibliotecas: `nltk`, `psutil`

```bash
pip install nltk psutil
```

Os pacotes do NLTK necessários (`stopwords`, `punkt`, `punkt_tab`) são baixados automaticamente na primeira execução, caso não estejam disponíveis localmente.

## Uso

### Indexação

```bash
python3 indexer.py -m <memoria> -c <corpus> -i <indice>
```

- `-m`: memória máxima disponível para o indexador, em megabytes (referência mínima: 1024).
- `-c`: caminho para o arquivo do corpus (JSONL) a ser indexado.
- `-i`: caminho para o diretório onde os arquivos de índice serão gravados.

Ao final, imprime em stdout um JSON com estatísticas do índice gerado:

```json
{
    "Index Size": 0.0,
    "Elapsed Time": 0.0,
    "Number of Lists": 0,
    "Average List Size": 0.0
}
```

### Processamento de consultas

```bash
python3 processor.py -i <indice> -q <consultas> -r <TFIDF|BM25>
```

- `-i`: caminho para o diretório do índice gerado pelo `indexer.py`.
- `-q`: caminho para o arquivo de consultas (uma consulta por linha).
- `-r`: função de ranqueamento, `TFIDF` ou `BM25`.

Para cada consulta, imprime em stdout um JSON no formato:

```json
{
  "Query": "information retrieval",
  "Results": [
    { "ID": "0512698", "Score": 24.2 },
    { "ID": "0249777", "Score": 12.4 }
  ]
}
```

## Políticas implementadas

### Indexador

- **Pré-processamento**: tokenização, remoção de stopwords e stemming (Porter Stemmer), aplicados a título e corpo de cada documento.
- **Gerenciamento de memória**: leitura do corpus em lotes; o uso de memória do processo é monitorado com `psutil` e, ao ultrapassar o limiar configurado, os índices parciais acumulados em memória são descarregados em disco (`salvar_indices_disco`).
- **Paralelização**: pré-processamento dos documentos de cada lote distribuído entre múltiplas threads (`ThreadPoolExecutor`).
- **Merge externo**: os índices parciais gravados em disco são mesclados via *external mergesort* (`external_mergesort`), usando `heapq.merge` sobre geradores que leem cada arquivo parcial sequencialmente, produzindo o índice invertido final ordenado, o léxico de termos e a distribuição de tamanhos das listas.
- **Artefatos gerados**: `indice_invertido_final.jsonl`, `documentos_final.jsonl`, `lexico.jsonl` e `distribuicao_tamanhos.csv`.

### Processador de consultas

- **Pré-processamento**: mesma pipeline do indexador (tokenização, remoção de stopwords, stemming), aplicada às consultas.
- **Correspondência (matching)**: interseção conjuntiva *document-at-a-time* (DAAT) entre as listas invertidas dos termos da consulta.
- **Pontuação**: duas funções de ranqueamento disponíveis:
  - **TFIDF**: `freq(termo, doc) * log(N / df(termo))`, somado sobre os termos da consulta.
  - **BM25**: variante Okapi BM25, com `k=1.2` e `b=0.75`, considerando o tamanho do documento em relação ao tamanho médio do corpus.
- **Paralelização**: as consultas são processadas em paralelo com `ThreadPoolExecutor`.
- Os 10 documentos com maior score são selecionados por meio de uma heap, mantendo apenas os *top-k* durante a varredura.

## Corpus

O corpus utilizado nos experimentos está disponível em [Google Drive](https://drive.google.com/drive/folders/1s3EYLvNWpZwnY214ui07QWp5ZY0mMXNm?usp=sharing) (ver `Link do Drive.txt`).

## Enunciado

O enunciado completo do trabalho está disponível em [`pa2.pdf`](pa2.pdf).
