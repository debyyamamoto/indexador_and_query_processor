"""
p2 - recuperação da informação 
goal: implementar módulos de indexação e processamento de consultas de um mecanismo de busca na web.
entrega: código + caracterização do índice de tamanho médio + resultados obtidos para um conjunto de consultas

documento processor.py:

$ python3 processor.py -i <INDEX> -q <QUERIES> -r <RANKER>

- execução: $ python3 processor.py -i <index> -q <queries> -r <ranking>
- informações impressas em  um JSON após cada consulta:
{ 
  "Query": "information retrieval",
  "Results": [
    { "ID": "0512698", "Score": 24.2 },
    { "ID": "0249777", "Score": 12.4 }
  ] 
}

- a lista de resultados dever ser ordenada em ordem decrescente de pontuação do documento e incluir até os 10 principais resultados para aquela consulta

- Políticas de Processamento de Consultas:
    -  Para cada consulta na lista fornecida via argumento -q, sua implementação deve:
        - pré-processar a consulta, 
        - recuperar documentos candidatos a partir do índice fornecido (argumento -i), 
        - pontuar esses documentos de acordo com o modelo de ranqueamento escolhido (argumento -r) e 
        - imprimir os 10 principais resultados usando o formato mencionado acima. 
    Além desse fluxo de trabalho padrão, sua implementação deve seguir as seguintes políticas:
        IMPLEMENTADO - 1. Política de Pré-processamento: remoção de stopwords e stemming (e outras opcionais) (MESMA DO INDEXADOR)
        IMPLEMENTADO - 2. Política de Correspondência: Para maior eficiência, sua implementação deve realizar uma correspondência conjuntiva document-at-a-time (DAAT) ao recuperar documentos candidatos.
        IMPLEMENTADO - 3. Política de Pontuação: Sua implementação deve fornecer duas funções de pontuação: TFIDF e BM25. Você é livre para experimentar diferentes variantes dessas funções presentes na literatura.
        4. Política de Paralelização (extra): Para garantir a máxima eficiência, você pode paralelizar o processamento de consultas em várias threads. Você pode experimentar para encontrar um número ideal de threads para maximizar seu rendimento (throughput) enquanto minimiza a sobrecarga de paralelização gerada.
"""
import argparse
import os
import nltk
import json
import heapq
import math
import time


from functools import partial
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = 4

def configurar_nltk():
    """
    Tenta localizar os pacotes localmente. 
    Só faz o download (de forma silenciosa) se for a primeira vez rodando.
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
    parser = argparse.ArgumentParser(prog="Processor", description="Processa uma query e retorna documentos rankeados", epilog = "[]")
    parser.add_argument("-i", "--index", required=True, help = 'Caminho repositório para o Índice')
    parser.add_argument("-q", "--queries", required=True, help = 'Caminho para Arquivo de Consultas')
    parser.add_argument("-r", "--ranker", required=True, choices=["TFIDF", "BM25"], help='Função de ranqueamento: "TFIDF" ou "BM25"')
    args = parser.parse_args()
    return args.index, args.queries, args.ranker

def carregamento_info_docs(i):
    """
    Carrega informações sobre os documentos 
    """
    tamanho_documento = {}
    soma_tamanhos = 0
    total_documentos = 0
    caminho_doc_final = os.path.join(i, "documentos_final.jsonl")
    with open(caminho_doc_final, 'r', encoding='utf-8') as f:
        for linha in f:
            dicionario_linha = json.loads(linha)
            docid = list(dicionario_linha.keys())[0]
            tamanho = dicionario_linha[docid]

            tamanho_documento[docid] = tamanho
            soma_tamanhos += tamanho
            total_documentos += 1

    media = soma_tamanhos / total_documentos if total_documentos > 0 else 0
    return tamanho_documento, total_documentos, media

def pre_processamento(query, stopWords):
    """
    Usar threads para realizar o pré- processamento nos arquivos: remoção de stopwords e stemming
    """
    # remover stopwords do título e do texto
    query_limpa = [token for token in query if (token.lower() not in stopWords and token.isalnum())]
    # aplicar stemming
    radicais_query = [stemmer.stem(palavra) for palavra in query_limpa]
    return (radicais_query)

def processamento_query(query, i):
    """
    Retorna o Dicionario contendo a lista Invertida de Cada Token da Query, se a lista invertida global não contiver alguma palavra da query, retorna uma lista vazia
    """
    # abrir documento query
    tokens_query = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    
    radicais_query = pre_processamento(tokens_query, stopWords=stop_words)
    if not radicais_query:
        return []
    termos_alvo = set(radicais_query)
    token_posts = {}
    caminho_final = os.path.join(i, "indice_invertido_final.jsonl")
    # abrir indice 
    with open(caminho_final, 'r', encoding='utf-8') as f:
        for linha in f:
            if any(f'"{termo}":' in linha for termo in termos_alvo):
                dicionario = json.loads(linha)
                termo = list(dicionario.keys())[0]
                if termo in termos_alvo:
                    token_posts[termo] = dicionario[termo]

                    if len(token_posts) == len(termos_alvo):
                        break
    # percorrer os tokens da query e carregar seus tokens para a RAM
    if len(token_posts) < len(termos_alvo):
            return {}
    return token_posts

def daat(token_posts):
    """
    Realiza o Matching de Documentos dos Termos de uma Query seguindo Document at a Time
    """
    if not token_posts:
        return [], []
    # criar ponteiros para cada uma das listas
    listas = list(token_posts.values())
    num_termos = len(listas)

    lista_ponteiros = [0] * num_termos
    resultados_encontrados = []

    while all(lista_ponteiros[i] < len(listas[i]) for i in range(num_termos)):
        #ai, vamos colocar em um set os docids atuais, se o tamanho do set for 1, um documento válido foi encontrado 
        docs_atuais = []
        for i in range(num_termos):
            docs_atuais.append(listas[i][lista_ponteiros[i]][0])
        if len(set(docs_atuais)) == 1:
            frequencia = []
            for i in range(num_termos):
                frequencia.append(listas[i][lista_ponteiros[i]][1])
            resultados_encontrados.append((docs_atuais[0], frequencia))
            for i in range(num_termos):
                lista_ponteiros[i] += 1
        else:
            maior_doc = float('-inf')
            for i in range(num_termos):
                doc_atual_int = int(listas[i][lista_ponteiros[i]][0])
                if doc_atual_int > maior_doc:
                    maior_doc = doc_atual_int
            for i in range(num_termos):
                doc_atual_int = int(listas[i][lista_ponteiros[i]][0])
                if doc_atual_int < maior_doc:
                    lista_ponteiros[i] += 1
    return resultados_encontrados, listas

def tfidf(resultados_encontrados, total_documentos, listas):
    """
    Rankeamento baseado em quantas vezes cada palavra aparece
    """
    heap = []
    todos_os_scores = []
    for docid, frequencias in resultados_encontrados:
        score_total = 0
        for i in range(len(frequencias)):
            freq_doc = frequencias[i]
            df = len(listas[i])
            score = freq_doc * math.log(total_documentos/df)
            score_total += score
        heapq.heappush(heap, (score_total, docid))
        todos_os_scores.append(score_total)
        if len(heap) > 10:
            heapq.heappop(heap)
    return sorted(heap, reverse=True), todos_os_scores

def bm25(resultados_encontrados, total_documentos, listas,  media, tamanho_documento, k, b):
    """
    Rankeamento que considera Saturação e Penalização por Tamanho
    """
    heap = []
    todos_os_scores = []
    for docid, frequencias in resultados_encontrados:
        score_total = 0
        for i in range(len(frequencias)):
            freq_doc = frequencias[i]
            df = len(listas[i])
            idf = math.log(((total_documentos- df + 0.5)/(df + 0.5)) + 1)
            if media > 0:
                score_total += idf * (freq_doc * (k + 1))/ (freq_doc + k * (1 - b + b * (tamanho_documento[docid]/media)))
            else:
                score_total += 0
        heapq.heappush(heap, (score_total, docid))
        todos_os_scores.append(score_total)
        if len(heap) > 10:
            heapq.heappop(heap)
    return sorted(heap, reverse=True), todos_os_scores

def rankeamento(resultados_encontrados, modelo_rankeamento, total_documentos, listas, media, tamanho_documento, k, b):
    """
    Gerencia a execução da política de rankeamento dependendo da escolha do usuário
    """
    if modelo_rankeamento == "TFIDF":
        ranking, todos_os_scores = tfidf(resultados_encontrados, total_documentos, listas)
    else:
        ranking, todos_os_scores = bm25(resultados_encontrados, total_documentos, listas,  media, tamanho_documento, k, b)
    return ranking, todos_os_scores

def worker_processar_query(query, i, r, total_documentos, media, tamanho_documento):
    dicio = processamento_query(query, i)
    resultados_encontrados, listas = daat(dicio)
    
    num_matches = len(resultados_encontrados)
    todos_scores = [] # INICIALIZAÇÃO PARA EVITAR O ERRO
    
    if resultados_encontrados:
        ranking_bruto, todos_scores = rankeamento(resultados_encontrados, r, total_documentos, listas, media, tamanho_documento, k=1.2, b=0.75)
    else:
        ranking_bruto = []
        
    resultados_formatados = []
    for score, docid in ranking_bruto:
        resultados_formatados.append({
            "ID": docid,
            "Score": round(score, 2) 
        })
        
    return {
        "Query": query,
        "Results": resultados_formatados,
        "Stats": {
            "Num_Matches": num_matches,
            "All_Scores": todos_scores
        }
    }
if __name__ == "__main__":
    stemmer = PorterStemmer()
    configurar_nltk()
    i, q, r = parser_argumentos()
    tamanho_documento, total_documentos, media = carregamento_info_docs(i)

    queries = []
    with open(q, 'r', encoding='utf-8') as f_queries:
        for linha in f_queries:
            query_texto = linha.strip()
            if not query_texto:
                continue
            else:
                queries.append(query_texto)
            
    func_parcial = partial(worker_processar_query, i=i, r=r, total_documentos=total_documentos, media=media, tamanho_documento=tamanho_documento)
    inicio = time.time()
    with ThreadPoolExecutor(NUM_THREADS) as executor:
        resultados_finais = list(executor.map(func_parcial, queries))
    tempo_total = time.time() - inicio
    stats_para_relatorio = {}
    todas_as_distribuicoes = []
    for saida in resultados_finais:
        # Extrai os status (não queremos que o usuário final veja isso no terminal)
        stats = saida.pop("Stats")
        
        stats_para_relatorio[saida["Query"]] = stats["Num_Matches"]
        todas_as_distribuicoes.extend(stats["All_Scores"])
        
        # Imprime o formato exigido pelo professor
        print(json.dumps(saida, indent=2))
    with open(f"matches_por_query_{r}.json", "w") as f:
        json.dump({"Tempo_Total_Segundos": tempo_total, "Matches": stats_para_relatorio}, f, indent=4)
        
    with open(f"scores_distribuicao_{r}.json", "w") as f:
        json.dump(todas_as_distribuicoes, f)

    for saida_json in resultados_finais:
        print(json.dumps(saida_json, indent=2))