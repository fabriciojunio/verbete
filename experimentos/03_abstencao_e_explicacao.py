"""Experimento 3: quando o modelo deve calar, e como ele explica o que decidiu.

Num produto de monitoramento regulatorio, classificar errado custa mais do que
nao opinar: quem recebe a classificacao age sobre ela, e um projeto de lei que
foi para a mesa errada so e descoberto tarde. Entao a pergunta de produto nao e
"qual o F1", e sim **"com que cobertura eu consigo que F1"**.

A segunda parte e a explicacao. O classificador e linear de proposito: os
termos que mais pesam saem do proprio modelo que decide, e nao de uma
aproximacao posterior. Num setor regulado, "o modelo disse" nao e resposta, e
uma explicacao que nao vem do modelo que decidiu e so uma segunda opiniao.

Uso:
    python experimentos/03_abstencao_e_explicacao.py
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    ap.add_argument(
        "--f1-alvo",
        type=float,
        default=0.70,
        help="o micro-F1 que o produto precisa entregar; o script responde com "
        "que cobertura ele e alcancado",
    )
    args = ap.parse_args()

    saida = Path(args.saida)
    for nome in ("escores_teste_sem_keywords.npy", "y_teste.npy", "treino.json"):
        if not (saida / nome).exists():
            print(f"falta {saida / nome}: rode 02_treinar.py primeiro",
                  file=sys.stderr)
            return 1

    escores = np.load(saida / "escores_teste_sem_keywords.npy")
    y = np.load(saida / "y_teste.npy")
    treino = json.loads((saida / "treino.json").read_text(encoding="utf-8"))
    limiares = np.asarray(
        treino["resultados"]["sem_keywords"]["limiares"], dtype=float
    )
    print(f"{y.shape[0]} proposicoes de teste, {y.shape[1]} temas")

    # --------------------------------------------------- a curva de abstencao
    # o limiar de decisao e o vetor por rotulo, mas a CONFIANCA da linha e o
    # maior escore: e a unica leitura que nao depende de qual rotulo disparou
    # A curva aqui e calculada com o vetor de limiares POR ROTULO, que e o do
    # modelo de verdade. `curva_de_abstencao` do dominio aceita um limiar unico
    # e serve ao caso geral; aqui a decisao ja esta tomada por rotulo e o que
    # varia e so o corte de confianca da linha.
    from verbete.dominio.avaliacao import acerto_exato, macro, micro

    previsto = escores >= limiares
    confianca = escores.max(axis=1)
    cortes = tuple(np.round(np.quantile(confianca, np.linspace(0, 0.9, 10)), 4))
    linhas = []
    for corte in cortes:
        atende = confianca >= corte
        if not atende.any():
            continue
        linhas.append(
            {
                "corte_de_confianca": float(corte),
                "cobertura": float(atende.mean()),
                "n_respondidos": int(atende.sum()),
                "micro_f1": micro(y[atende], previsto[atende])["f1"],
                "macro_f1": macro(y[atende], previsto[atende])["f1"],
                "acerto_exato": acerto_exato(y[atende], previsto[atende]),
            }
        )

    print(f"\n{'corte':>8} {'cobertura':>10} {'respondidos':>12} "
          f"{'micro-F1':>9} {'macro-F1':>9} {'exato':>7}")
    for x in linhas:
        print(f"{x['corte_de_confianca']:>8.3f} {x['cobertura']:>10.1%} "
              f"{x['n_respondidos']:>12} {x['micro_f1']:>9.3f} "
              f"{x['macro_f1']:>9.3f} {x['acerto_exato']:>7.3f}")

    pd.DataFrame(linhas).to_csv(saida / "abstencao.csv", index=False)

    base = linhas[0]
    atinge = [x for x in linhas if x["micro_f1"] >= args.f1_alvo]
    print("\n== a leitura de produto ==")
    print(f"  respondendo tudo: micro-F1 de {base['micro_f1']:.3f} em 100% dos casos")
    if atinge:
        melhor = max(atinge, key=lambda x: x["cobertura"])
        print(f"  para micro-F1 de {args.f1_alvo:.2f}: cobertura de "
              f"{melhor['cobertura']:.1%}, ou seja "
              f"{100 - melhor['cobertura'] * 100:.0f}% vao para revisao humana")
        leitura = {
            "f1_alvo": args.f1_alvo,
            "cobertura_no_alvo": melhor["cobertura"],
            "corte_no_alvo": melhor["corte_de_confianca"],
        }
    else:
        print(f"  micro-F1 de {args.f1_alvo:.2f} NAO e alcancado em cobertura "
              f"nenhuma. O melhor da curva e "
              f"{max(x['micro_f1'] for x in linhas):.3f}.")
        print("  Essa e a resposta de produto: com este modelo, o corte de")
        print("  confianca nao compra qualidade suficiente, e o caminho e")
        print("  melhorar o modelo e nao apertar o corte.")
        leitura = {
            "f1_alvo": args.f1_alvo,
            "cobertura_no_alvo": None,
            "melhor_f1_da_curva": max(x["micro_f1"] for x in linhas),
        }

    ganho = max(x["micro_f1"] for x in linhas) - base["micro_f1"]
    print(f"\n  ganho maximo de micro-F1 ao abster: {ganho:+.3f}")
    if ganho < 0.02:
        print("  Ganho pequeno significa que o escore do modelo NAO separa o")
        print("  que ele acerta do que ele erra. Isso e informacao sobre o")
        print("  modelo, e nao sobre a tarefa: um escore que nao ordena acerto")
        print("  nao serve para decidir quando calar.")

    # ----------------------------------------------------------- a explicacao
    termos = json.loads((saida / "termos_por_tema.json").read_text(encoding="utf-8"))
    por_tema = pd.read_csv(saida / "por_tema.csv")
    melhores = por_tema.nlargest(5, "f1")
    piores = por_tema[por_tema["apoio"] >= 5].nsmallest(5, "f1")

    print("\n== os temas que o modelo acerta, e o que pesa neles ==")
    for r in melhores.itertuples():
        a = termos.get(r.rotulo, {}).get("a_favor", [])[:6]
        print(f"  {r.rotulo[:44]:<44} F1={r.f1:.2f} apoio={r.apoio}")
        print(f"      {', '.join(x['termo'] for x in a)}")

    print("\n== os temas que ele erra, e o que ele estava olhando ==")
    for r in piores.itertuples():
        a = termos.get(r.rotulo, {}).get("a_favor", [])[:6]
        print(f"  {r.rotulo[:44]:<44} F1={r.f1:.2f} apoio={r.apoio}")
        print(f"      {', '.join(x['termo'] for x in a)}")

    (saida / "abstencao.json").write_text(
        json.dumps(
            {
                "curva": linhas,
                "leitura_de_produto": leitura,
                "ganho_maximo_ao_abster": ganho,
                "temas_que_acerta": melhores["rotulo"].tolist(),
                "temas_que_erra": piores["rotulo"].tolist(),
                "observacao_sobre_a_explicacao": (
                    "os termos sao os coeficientes do proprio classificador que "
                    "decide, nao uma aproximacao posterior. Num setor regulado, "
                    "explicacao que nao vem do modelo que decidiu e so uma "
                    "segunda opiniao"
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nartefatos em {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
