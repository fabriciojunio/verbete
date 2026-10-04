"""Experimento 4: as figuras, todas geradas por script.

Nenhuma figura e feita a mao nem editada depois. Figura editada deixa de
corresponder ao dado no momento em que o dado muda, e a unica forma de garantir
que a figura e o numero contam a mesma historia e as duas sairem do mesmo
arquivo.

Uso:
    python experimentos/04_figuras.py
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

ESCURO = "#22303f"
QUENTE = "#b06a3b"


def _salvar(fig, destino: Path, nome: str) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for extensao in ("png", "pdf"):
        fig.savefig(destino / f"{nome}.{extensao}")
    plt.close(fig)
    print(f"  ok   {nome}")


def figura_modelos(saida: Path, figuras: Path) -> bool:
    caminho = saida / "modelos.csv"
    if not caminho.exists():
        return False
    d = pd.read_csv(caminho)
    d = d[(d["conjunto"] == "teste") & (d["variante"] == "sem_keywords")]
    if d.empty:
        return False
    ordem = ["sempre_os_frequentes", "por_palavra_chave", "tfidf_linear"]
    d = d.set_index("modelo").reindex([m for m in ordem if m in set(d["modelo"])])

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(d))
    altura = 0.38
    ax.barh(y + altura / 2, d["micro_f1"], altura, label="micro-F1", color=ESCURO)
    ax.barh(y - altura / 2, d["macro_f1"], altura, label="macro-F1", color=QUENTE)
    for i, (mi, ma) in enumerate(zip(d["micro_f1"], d["macro_f1"], strict=True)):
        ax.text(mi, i + altura / 2, f" {mi:.3f}", va="center", fontsize=8)
        ax.text(ma, i - altura / 2, f" {ma:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [s.replace("_", " ") for s in d.index], fontsize=8
    )
    ax.set_xlabel("F1 no conjunto de teste")
    ax.set_xlim(0, max(d["micro_f1"].max(), d["macro_f1"].max()) * 1.22)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(
        "A distância entre micro e macro é o resultado:\nela mede o quanto o"
        " modelo só funciona nos temas comuns",
        loc="left",
        fontsize=9,
    )
    _salvar(fig, figuras, "modelos")
    return True


def figura_vazamento(saida: Path, figuras: Path) -> bool:
    caminho = saida / "treino.json"
    if not caminho.exists():
        return False
    d = json.loads(caminho.read_text(encoding="utf-8")).get("vazamento_de_anotacao")
    if not d:
        return False

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    rotulos = ["micro-F1", "macro-F1"]
    sem = [d["micro_f1_sem"], d["macro_f1_sem"]]
    com = [d["micro_f1_com"], d["macro_f1_com"]]
    y = np.arange(2)
    ax.barh(y + 0.19, com, 0.36, label="com as palavras-chave da indexação", color=QUENTE)
    ax.barh(y - 0.19, sem, 0.36, label="sem elas (o que existe em produção)", color=ESCURO)
    for i in range(2):
        ax.text(com[i], i + 0.19, f"  {com[i]:.3f}", va="center", fontsize=8)
        ax.text(sem[i], i - 0.19, f"  {sem[i]:.3f}", va="center", fontsize=8)
        ax.annotate(
            f"+{com[i] - sem[i]:.3f}",
            xy=(max(com[i], sem[i]) + 0.09, i),
            va="center", fontsize=8, color=QUENTE, fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos)
    ax.set_xlim(0, max(com + sem) * 1.42)
    ax.set_xlabel("F1 no conjunto de teste")
    # a legenda vai acima do eixo: dentro do gráfico ela cobre o rótulo da barra
    ax.legend(
        frameon=False, fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, -0.42), ncol=2,
    )
    ax.set_title("O tamanho do vazamento de anotação", loc="left")
    _salvar(fig, figuras, "vazamento")
    return True


def figura_abstencao(saida: Path, figuras: Path) -> bool:
    caminho = saida / "abstencao.csv"
    if not caminho.exists():
        return False
    d = pd.read_csv(caminho).sort_values("cobertura")
    if len(d) < 3:
        return False

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(d["cobertura"], d["micro_f1"], "o-", lw=1.5, ms=4,
            color=ESCURO, label="micro-F1")
    ax.plot(d["cobertura"], d["macro_f1"], "s--", lw=1.2, ms=3.5,
            color=QUENTE, label="macro-F1")
    ax.set_xlabel("cobertura: fração das proposições em que o modelo opina")
    ax.set_ylabel("F1 entre as respondidas")
    ax.set_xlim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title(
        "O que se ganha deixando o modelo calar quando está inseguro",
        loc="left",
        fontsize=9,
    )
    _salvar(fig, figuras, "abstencao")
    return True


def figura_por_tema(saida: Path, figuras: Path) -> bool:
    caminho = saida / "por_tema.csv"
    if not caminho.exists():
        return False
    d = pd.read_csv(caminho)
    d = d[d["apoio"] > 0].sort_values("apoio", ascending=True).tail(22)
    if len(d) < 4:
        return False

    fig, ax = plt.subplots(figsize=(6.8, max(3.0, 0.27 * len(d) + 1.2)))
    cores = [ESCURO if f >= 0.5 else QUENTE for f in d["f1"]]
    ax.barh(np.arange(len(d)), d["f1"], color=cores, height=0.7)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(
        [f"{r.rotulo[:38]} ({r.apoio})" for r in d.itertuples()], fontsize=7
    )
    ax.set_xlabel("F1 por tema, no conjunto de teste")
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "O número entre parênteses é quantos casos o tema tem.\nO modelo"
        " funciona onde há exemplos, e é isso que o macro-F1 cobra.",
        loc="left",
        fontsize=9,
    )
    _salvar(fig, figuras, "por_tema")
    return True


def figura_distribuicao(saida: Path, figuras: Path) -> bool:
    caminho = saida / "treino.json"
    if not caminho.exists():
        return False
    d = json.loads(caminho.read_text(encoding="utf-8")).get("distribuicao_dos_temas")
    if not d:
        return False
    s = pd.Series(d).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6.6, max(3.0, 0.26 * len(s) + 1.0)))
    ax.barh(np.arange(len(s)), s.to_numpy(), color=ESCURO, height=0.7)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels([x[:40] for x in s.index], fontsize=7)
    ax.set_xlabel("proposições com o tema")
    razao = s.max() / max(s.min(), 1)
    ax.set_title(
        f"A taxonomia é desigual: o tema mais comum tem {razao:.0f}x\nmais casos"
        " que o mais raro",
        loc="left",
        fontsize=9,
    )
    _salvar(fig, figuras, "distribuicao_dos_temas")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = Path(args.saida)
    figuras = saida / "figuras"

    feitas = 0
    for nome, f in (
        ("modelos", figura_modelos),
        ("vazamento", figura_vazamento),
        ("abstencao", figura_abstencao),
        ("por_tema", figura_por_tema),
        ("distribuicao_dos_temas", figura_distribuicao),
    ):
        if f(saida, figuras):
            feitas += 1
        else:
            print(f"  pula {nome}: falta o artefato que ela desenha")

    if feitas == 0:
        print("nenhuma figura gerada: rode os experimentos anteriores",
              file=sys.stderr)
        return 1
    print(f"\n{feitas} figuras em {figuras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
