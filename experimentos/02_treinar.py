"""Experimento 2: treina os candidatos, mede o vazamento de anotacao e escolhe.

Tres coisas acontecem aqui, e a segunda e a que o projeto existe para mostrar:

1. os tres candidatos sao treinados no mesmo recorte e medidos nas mesmas
   metricas, com a linha de base sem aprendizado em pe de igualdade;
2. **o vazamento de anotacao e medido, nao discutido.** O campo `keywords` e
   preenchido pela mesma indexacao humana que atribui o tema. O modelo e
   treinado com e sem ele, e a diferenca entre os dois numeros e o tamanho do
   vazamento. Reportar so o numero com keywords seria reportar um modelo que
   nao existe em producao, porque la a proposicao chega sem indexacao;
3. os limiares por rotulo sao ajustados **na validacao**, e so depois o teste e
   tocado, uma unica vez.

A divisao e temporal: os anos anteriores treinam, o ultimo testa.

Uso:
    python experimentos/02_treinar.py --ano-de-teste 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from verbete.aplicacao.classificador import (  # noqa: E402
    PorPalavraChave,
    SempreOsFrequentes,
    TfidfLinear,
    matriz_de_rotulos,
    montar_texto,
)
from verbete.dominio.avaliacao import (  # noqa: E402
    avaliar,
    melhor_limiar_por_rotulo,
)


def _preparar(registros, temas, usar_keywords):
    textos = [montar_texto(r, usar_keywords=usar_keywords) for r in registros]
    y = matriz_de_rotulos(registros, temas)
    return textos, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano-de-teste", type=int, default=2024)
    ap.add_argument("--fracao-de-validacao", type=float, default=0.2)
    ap.add_argument("--semente", type=int, default=20261004)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()

    saida = Path(args.saida)
    caminho = saida / "proposicoes.json"
    if not caminho.exists():
        print(f"falta {caminho}: rode 01_coletar.py primeiro", file=sys.stderr)
        return 1

    registros = json.loads(caminho.read_text(encoding="utf-8"))
    cartao = json.loads((saida / "cartao_do_dado.json").read_text(encoding="utf-8"))
    temas = cartao["taxonomia"]["temas"]
    print(f"{len(registros)} proposicoes, {len(temas)} temas oficiais")

    com_tema = [r for r in registros if r.get("temas")]
    print(f"  com ao menos um tema: {len(com_tema)}")
    if len(com_tema) < 300:
        print("amostra pequena demais para treinar e reportar", file=sys.stderr)
        return 1

    # ------------------------------------------------- divisao temporal
    anteriores = [r for r in com_tema if int(r["ano"]) < args.ano_de_teste]
    teste = [r for r in com_tema if int(r["ano"]) == args.ano_de_teste]
    if not anteriores or not teste:
        print(
            f"divisao temporal impossivel: {len(anteriores)} antes de "
            f"{args.ano_de_teste} e {len(teste)} nele",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(args.semente)
    ordem = rng.permutation(len(anteriores))
    corte = int(len(anteriores) * (1 - args.fracao_de_validacao))
    treino = [anteriores[i] for i in ordem[:corte]]
    validacao = [anteriores[i] for i in ordem[corte:]]
    print(f"  treino {len(treino)} | validacao {len(validacao)} | "
          f"teste {len(teste)} (ano {args.ano_de_teste})")

    distribuicao = pd.Series(
        [t for r in com_tema for t in r["temas"]]
    ).value_counts()
    print("\n== os temas, do mais comum ao mais raro ==")
    print(f"  {len(distribuicao)} temas aparecem; o mais comum tem "
          f"{distribuicao.iloc[0]} casos e o mais raro tem {distribuicao.iloc[-1]}")
    for nome, n in distribuicao.head(8).items():
        print(f"    {nome[:48]:<48} {n}")
    print(f"    ... e {len(distribuicao) - 8} temas com menos casos")

    # ------------------------------------- com e sem o vazamento de anotacao
    resultados = {}
    for usar_keywords in (False, True):
        rotulo = "com_keywords" if usar_keywords else "sem_keywords"
        print(f"\n{'=' * 62}\n== {rotulo} ==")
        x_tr, y_tr = _preparar(treino, temas, usar_keywords)
        x_va, y_va = _preparar(validacao, temas, usar_keywords)
        x_te, y_te = _preparar(teste, temas, usar_keywords)

        candidatos = {
            "sempre_os_frequentes": SempreOsFrequentes(),
            "por_palavra_chave": PorPalavraChave(temas=temas),
            "tfidf_linear": TfidfLinear(),
        }
        deste = {}
        for nome, m in candidatos.items():
            m.ajustar(x_tr, y_tr)
            if isinstance(m, TfidfLinear):
                # limiar por rotulo, ajustado NA VALIDACAO. Limiar unico para 32
                # temas de frequencias muito diferentes e simplicidade falsa
                limiares = melhor_limiar_por_rotulo(y_va, m.escores(x_va))
                p_va = m.escores(x_va) >= limiares
                p_te = m.escores(x_te) >= limiares
                deste["limiares"] = limiares.tolist()
            else:
                p_va = m.prever(x_va)
                p_te = m.prever(x_te)
            deste[nome] = {
                "validacao": avaliar(y_va, p_va),
                "teste": avaliar(y_te, p_te, nomes=temas),
            }

        print(f"{'modelo':>22} {'micro-F1':>9} {'macro-F1':>9} "
              f"{'exato':>7} {'um certo':>9}")
        for nome in candidatos:
            t = deste[nome]["teste"]
            print(f"{nome:>22} {t['micro_f1']:>9.3f} {t['macro_f1']:>9.3f} "
                  f"{t['acerto_exato']:>7.3f} {t['ao_menos_um_certo']:>9.3f}")
        resultados[rotulo] = deste

        if usar_keywords is False:
            modelo_sem = candidatos["tfidf_linear"]
            np.save(saida / "escores_teste_sem_keywords.npy", modelo_sem.escores(x_te))
            np.save(saida / "y_teste.npy", y_te)
            modelo_guardado = modelo_sem

    # ------------------------------------------------- o tamanho do vazamento
    sem = resultados["sem_keywords"]["tfidf_linear"]["teste"]
    com = resultados["com_keywords"]["tfidf_linear"]["teste"]
    vazamento = {
        "micro_f1_sem": sem["micro_f1"],
        "micro_f1_com": com["micro_f1"],
        "diferenca_micro": com["micro_f1"] - sem["micro_f1"],
        "macro_f1_sem": sem["macro_f1"],
        "macro_f1_com": com["macro_f1"],
        "diferenca_macro": com["macro_f1"] - sem["macro_f1"],
    }
    print(f"\n{'=' * 62}")
    print("== o tamanho do vazamento de anotacao ==")
    print(f"  micro-F1: {sem['micro_f1']:.3f} sem keywords, "
          f"{com['micro_f1']:.3f} com ({vazamento['diferenca_micro']:+.3f})")
    print(f"  macro-F1: {sem['macro_f1']:.3f} sem keywords, "
          f"{com['macro_f1']:.3f} com ({vazamento['diferenca_macro']:+.3f})")
    print("\n  As keywords sao preenchidas pela mesma indexacao humana que")
    print("  atribui o tema. Em producao a proposicao chega sem elas, entao o")
    print("  numero que vale e o SEM. O numero COM esta aqui para mostrar")
    print("  quanto um modelo pareceria melhor se ninguem tivesse perguntado")
    print("  de onde a variavel vem.")

    # ---------------------------------------- de quanto ganha de nao ter modelo
    piso = resultados["sem_keywords"]["sempre_os_frequentes"]["teste"]
    regra = resultados["sem_keywords"]["por_palavra_chave"]["teste"]
    ganho = {
        "micro_sobre_sempre_os_frequentes": (
            sem["micro_f1"] / piso["micro_f1"] if piso["micro_f1"] > 0 else None
        ),
        "micro_sobre_palavra_chave": (
            sem["micro_f1"] / regra["micro_f1"] if regra["micro_f1"] > 0 else None
        ),
        "macro_sobre_palavra_chave": (
            sem["macro_f1"] / regra["macro_f1"] if regra["macro_f1"] > 0 else None
        ),
    }
    print("\n== de quanto o modelo ganha de nao ter modelo ==")
    for k, v in ganho.items():
        print(f"  {k}: {v:.2f}x" if v else f"  {k}: nao definido")

    print("\n== distancia entre micro e macro, no modelo escolhido ==")
    print(f"  {sem['distancia_micro_macro']:.3f}")
    print("  Essa distancia e o resultado: ela mede o quanto o modelo esta")
    print("  funcionando so nos temas comuns. Reportar so o micro-F1 esconderia")
    print("  que a taxonomia inteira nao esta coberta.")

    (saida / "treino.json").write_text(
        json.dumps(
            {
                "divisao": {
                    "tipo": "temporal por ano",
                    "motivo": (
                        "proposicao sobre o mesmo assunto reaparece a cada "
                        "legislatura com ementa quase igual; divisao aleatoria "
                        "poria a quase-copia no treino e no teste"
                    ),
                    "ano_de_teste": args.ano_de_teste,
                    "treino": len(treino),
                    "validacao": len(validacao),
                    "teste": len(teste),
                },
                "distribuicao_dos_temas": distribuicao.to_dict(),
                "resultados": resultados,
                "vazamento_de_anotacao": vazamento,
                "ganho_sobre_nao_ter_modelo": ganho,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    linhas = []
    for rotulo, d in resultados.items():
        for nome, m in d.items():
            if nome == "limiares":
                continue
            for conjunto in ("validacao", "teste"):
                linhas.append(
                    {
                        "variante": rotulo,
                        "modelo": nome,
                        "conjunto": conjunto,
                        **{
                            k: v
                            for k, v in m[conjunto].items()
                            if k != "por_rotulo"
                        },
                    }
                )
    pd.DataFrame(linhas).to_csv(saida / "modelos.csv", index=False)

    pd.DataFrame(sem["por_rotulo"]).to_csv(saida / "por_tema.csv", index=False)

    # os termos que pesam, para o experimento de explicacao
    explicacao = {
        temas[j]: modelo_guardado.termos_que_pesam(j)
        for j in range(len(temas))
    }
    (saida / "termos_por_tema.json").write_text(
        json.dumps(explicacao, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nartefatos em {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
