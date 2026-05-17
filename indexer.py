"""
p2 - recuperação da informação 
goal: implementar módulos de indexação e processamento de consultas de um mecanismo de busca na web.
entrega: código + caracterização do índice de tamanho médio + resultados obtidos para um conjunto de consultas

documento indexer.py
$ python3 indexer.py -m <memory> -c <corpus> -i <index>
    -m <memory>: a memória disponível para o indexador em megabytes.
    -c <corpus>: o caminho para o arquivo do corpus a ser indexado.
    -i <index>: o caminho para o diretório onde os índices devem ser gravados.

- No final da execução, o documento deve imprimir um documento json na saída padrão (stdout) com as seguintes estatísticas:
    - Tamanho do Índice - em megabytes
    - Tempo decorrido para produzir o índice - em segundos
    - Número de Listas - número de listas invertidas no índice 
    - Tamanho médio da Lista - o número médio de postagens por lista invertida
- Políticas de Indexção 
    -  Para cada documento no corpus, sua implementação deve fazer o parse, tokenizá-lo e indexá-lo.
    - Sua implementação deve operar dentro do orçamento de memória designado (o argumento -m) durante toda a execução
    - No final devem ser armazenados como arquivos separarados:
        - Índice invertido - ok
        - Índice de documentos - ok
        - Lexico de termos - ok
- Se atentar: 
    IMPLEMENTADO - 1. Política de Pré Processamento - remoção de stopwords e stemming (e outras opcionais)
    IMPLEMENTADO - 2. Política de Gerenciamento de Memória - deve ser capaz de produzir índices parciais em memória (respeitando o orçamento de memória imposto) e mesclá-los (merge) em disco
    IMPLEMENTADO - 3. Política de Paralelização - paralelizar o processo de indexação em várias threads - Você pode experimentar para encontrar um número ideal de threads para minimizar o tempo de indexação, ao mesmo tempo em que minimiza a sobrecarga de paralelização gerada
    não - 4. Política de Compressão - Opcionalmente, você pode escolher implementar um esquema de compressão para as entradas do índice (ex: codificação gamma para docids, unária para frequência de termos) para máxima eficiência de armazenamento.

    - Observe que o orçamento de memória refere-se à memória total disponível para a sua implementação, e não apenas à memória necessária para armazenar as estruturas de índice em si. Como limite inferior de referência, assuma que a sua implementação será testada com -m 1024
    - O limite de memória será estritamente aplicado durante a avaliação. Se o seu programa excedê-lo, ele poderá ser automaticamente encerrado com um erro de falta de memória (OOM). Para evitar isso, use psutil.Process(os.getpid()).memory_info().rss para monitorar seu uso atual de memória (em bytes) e descarregue os índices parciais no disco antes de alocar mais memória, conforme necessário.

"""
import argparse
import os
import psutil
import nltk
import json
import heapq
import time
import logging
import sys

from functools import partial
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from collections import defaultdict

logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO,
    format='%(asctime)s - [INFO] - %(message)s',
    datefmt='%H:%M:%S'
)

MEMORY_THRESHOLD = 0.8
NUM_THREADS = 4
def configurar_nltk():
    """
    Tenta localizar os pacotes localmente.
    """
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

def parser_argumentos():
    """
    Função recebe instruções para a execução do índice
    """
    parser = argparse.ArgumentParser(prog="Indexador", description="Indexador de Corpus", epilog = "[]")
    parser.add_argument("-m", "--memory", type=int, required=True, help='Tamanho Máximo Memória RAM Permitida')
    parser.add_argument("-c", "--corpus", required=True, help='Caminho para o Corpus')
    parser.add_argument("-i", "--index", required=True, help='Caminho para o Índice')
    args = parser.parse_args()
    return args.memory, args.corpus, args.index

def leitura_corpus(m, corpus):
    """
    Função que mescla leitura em lotes + gerenciamento de memória
    """
    lote = []
    count = 0
    with open (corpus, 'r') as arquivo:
        for linha in arquivo:
            count += 1
            lote.append(linha)
            if count == 500:
                count = 0
                yield lote
                lote = []
    if lote:
        yield lote

def pre_processamento(linha, stopWords):
    """
    linhas divididas em: id, título e texto
    Usar threads para realizar o pré- processamento nos arquivos: remoção de stopwords e stemming
    """
    dicionario = json.loads(linha)
    doc_id = dicionario["id"]
    titulo = word_tokenize(dicionario["title"].lower())
    texto = word_tokenize(dicionario["text"].lower())
    # remover stopwords do título e do texto
    titulo_limpo = [token for token in titulo if (token.lower() not in stopWords and token.isalnum())]
    texto_limpo = [token for token in texto if (token.lower() not in stopWords and token.isalnum())]
    # aplicar stemming
    radicais_titulo = [stemmer.stem(palavra) for palavra in titulo_limpo]
    radicais_texto = [stemmer.stem(palavra) for palavra in texto_limpo]
    return (doc_id, radicais_titulo, radicais_texto)

def salvar_indices_disco(indice, pasta, prefixo, numero):
    """
    Função Responsável por Salvar os Índices Intermediários no Disco (Política de Gerenciamento de Memória)
    """
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(
        pasta,
        f"parcial_{prefixo}_{numero}.jsonl"
    )
    with open(caminho, "w", encoding="utf-8") as arquivo:
        for termo, postagens in indice.items():
            linha = json.dumps({termo: postagens})
            arquivo.write(linha + "\n")

def indexacao(m, c, i, total_documentos):
    """
    Função Principal de Indexação com Política de Paralelização 
    """
    stop_words = set(stopwords.words('english'))
    indice_invertido = defaultdict(list)
    indice_documentos = {}
    num_indices = 0
    with ThreadPoolExecutor(NUM_THREADS) as executor:
        for lote in leitura_corpus(m, c):
            func = partial(pre_processamento, stopWords=stop_words)
            resultados_do_lote = list(executor.map(func, lote))
            for docid, titulo, texto in resultados_do_lote:
                texto_completo = titulo + texto
                tamanho_documento = len(texto_completo)
                indice_documentos[docid] = tamanho_documento
                list_counts = Counter(texto_completo)
                for palavra in list_counts.items():
                    indice_invertido[palavra[0]].append((docid,palavra[1]))
            if indice_invertido:
                if psutil.Process(os.getpid()).memory_info().rss > MEMORY_THRESHOLD * m * 1000000:
                    num_indices += 1
                    salvar_indices_disco(dict(sorted(indice_invertido.items())), i,"indice", num_indices)
                    salvar_indices_disco(dict(sorted(indice_documentos.items())), i,"documentos", num_indices)
                    indice_invertido.clear()
                    indice_documentos.clear()
        if len(indice_invertido) > 0:
            num_indices += 1
            salvar_indices_disco(dict(sorted(indice_invertido.items())), i, "indice", num_indices)
            salvar_indices_disco(dict(sorted(indice_documentos.items())), i, "documentos", num_indices)
            indice_invertido.clear()
            indice_documentos.clear()

def unificar_documentos_e_limpar(i):
    """
    Função responsável por Retornar um Único arquivo de documentos - documentos_final - e limpa os arquivos parciais
    """
    caminho_doc_final = os.path.join(i, "documentos_final.jsonl")
    with open(caminho_doc_final, 'w', encoding='utf-8') as arquivo_final:
        for nome_arquivo in os.listdir(i):
            caminho_completo = os.path.join(i, nome_arquivo)
            if nome_arquivo.startswith("parcial_documentos_"):
                with open(caminho_completo, 'r', encoding='utf-8') as f_parcial:
                    for linha in f_parcial:
                        arquivo_final.write(linha)
                os.remove(caminho_completo)
            elif nome_arquivo.startswith("parcial_indice_"):
                os.remove(caminho_completo)

def transformar_tupla(ponteiro):
    """
    Transforma o ponteiro em um formato aceito pelo heapq
    """
    for linha in ponteiro:
        dicionario_linha = json.loads(linha)
        termo = list(dicionario_linha.keys())[0]
        postagens = dicionario_linha[termo]
        yield (termo, postagens)

def external_mergesort(i):
    """
    Realiza um mergesort externo no índice de termos invertidos e no arquivo de lexicos
    """
    arquivos_parciais = []
    # abrir os arquivos 
    for nome_arquivo in os.listdir(i):
        if nome_arquivo.startswith("parcial_indice_"):
            caminho_completo = os.path.join(i, nome_arquivo)
            arquivos_parciais.append(caminho_completo)
    ponteiros_arq = [open(arq, 'r', encoding='utf-8') for arq in arquivos_parciais]
    geradores = [transformar_tupla(p) for p in ponteiros_arq]
    caminho_final = os.path.join(i, "indice_invertido_final.jsonl")
    caminho_lexico = os.path.join(i, "lexico.jsonl")
    caminho_distribuicao = os.path.join(i, "distribuicao_tamanhos.csv")
    with open(caminho_final, 'w', encoding='utf-8') as arquivo_final, open(caminho_lexico, 'w', encoding='utf-8') as arquivo_lexico, open(caminho_distribuicao, 'w', encoding='utf-8') as arquivo_dist:
        termo_atual = None
        postagens_acumuladas = []
        count_termos = 0
        count_postagens = 0
        for termo, postagens in heapq.merge(*geradores):
            count_postagens += len(postagens)
            if termo == termo_atual:
                postagens_acumuladas.extend(postagens)
            else:
                if termo_atual is not None:
                    postagens_acumuladas.sort(key=lambda x: int(x[0]))
                    linha = json.dumps({termo_atual: postagens_acumuladas})
                    arquivo_final.write(linha + "\n")
                    arquivo_lexico.write(termo_atual + "\n")
                    arquivo_dist.write(f'"{termo_atual}",{len(postagens_acumuladas)}\n')
                count_termos += 1
                termo_atual = termo
                postagens_acumuladas = postagens
        if termo_atual is not None:
            postagens_acumuladas.sort(key=lambda x: int(x[0]))
            linha = json.dumps({termo_atual: postagens_acumuladas})
            arquivo_final.write(linha + "\n")
            arquivo_lexico.write(termo_atual + "\n")
            arquivo_dist.write(f'"{termo_atual}",{len(postagens_acumuladas)}\n')
    for p in ponteiros_arq:
        p.close()
    return caminho_final, count_termos, count_postagens

if __name__ == "__main__":
    configurar_nltk()
    stemmer = PorterStemmer()
    m, c, i = parser_argumentos()
    total_documentos = sum(1 for _ in open(c, 'rb')) 
    start = time.time()
    indexacao(m, c, i, total_documentos)
    caminho_final, count_termos, count_postagens = external_mergesort(i)
    unificar_documentos_e_limpar(i)
    finish = time.time()
    tempo_total = finish - start
    tamanho_arquivo = os.path.getsize(caminho_final)/(1024 * 1024)
    if count_termos > 0:
        tamanho_medio_listas = count_postagens/count_termos
    else:
        tamanho_medio_listas = 0

    estatisticas = {
        "Index Size": round(tamanho_arquivo, 2),
        "Elapsed Time": round(tempo_total, 2),
        "Number of Lists": count_termos,
        "Average List Size": round(tamanho_medio_listas, 2)
    }
    print(json.dumps(estatisticas, indent=4))
