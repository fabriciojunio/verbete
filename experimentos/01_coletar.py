"""Experimento 1: baixa as proposicoes e os temas atribuidos a elas.

O recorte e declarado na linha de comando e gravado no artefato. A divisao
treino/teste do projeto e **temporal**, por ano, e isso decide o recorte: e
preciso baixar anos consecutivos, e nao uma amostra de um ano so.

Por que temporal e nao aleatoria: proposicao sobre o mesmo assunto reaparece a
cada legislatura, muitas vezes com a ementa quase igual. Com divisao aleatoria,
a quase-copia de 2023 cai no treino e a de 2023 cai no teste, e o modelo parece
entender texto quando esta reconhecendo texto que ja viu. A divisao por ano
tambem e a que corresponde ao uso real: o modelo e treinado no passado e roda
sobre o que chega amanha.

Uso:
    python experimentos/01_coletar.py --anos 2022 2023 2024 --maximo-por-ano 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from verbete.adaptadores.camara import ClienteCamara, FalhaDaCamara  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--tipo", default="PL", help="PL = Projeto de Lei")
    ap.add_argument("--maximo-por-ano", type=int, default=1500)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()

    cli = ClienteCamara()
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    print("baixando a taxonomia oficial de temas...")
    try:
        temas = cli.temas_oficiais()
    except FalhaDaCamara as exc:
        print(f"nao foi possivel baixar os temas: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(temas)} temas")

    t0 = time.time()
    registros: list[dict] = []
    falhas: list[dict] = []
    por_ano: dict[int, int] = {}

    for ano in args.anos:
        print(f"\n== {ano} ==", flush=True)
        lista = cli.proposicoes(
            ano,
            sigla_tipo=args.tipo,
            maximo=args.maximo_por_ano,
            progresso=lambda ano, pagina, total: print(
                f"  pagina {pagina}: {total} proposicoes", flush=True
            ),
        )
        ids = [p["id"] for p in lista]
        print(f"  {len(ids)} proposicoes, buscando detalhe e temas...", flush=True)
        r, f = cli.coletar(
            ids,
            progresso=lambda feitos, total, falhas: print(
                f"    {feitos}/{total} ({feitos / total:.0%}), "
                f"{falhas} falhas, {(time.time() - t0) / 60:.1f} min",
                flush=True,
            ),
            aviso=lambda m: print(f"    aviso: {m}", flush=True),
        )
        registros.extend(r)
        falhas.extend(f)
        por_ano[ano] = len(r)

    minutos = (time.time() - t0) / 60
    com_tema = sum(1 for r in registros if r["temas"])
    print(f"\n{len(registros)} proposicoes em {minutos:.1f} min")
    print(f"  com ao menos um tema: {com_tema} ({com_tema / max(len(registros), 1):.1%})")
    print(f"  falhas: {len(falhas)}")

    if not registros:
        print("nada para gravar", file=sys.stderr)
        return 1

    (saida / "proposicoes.json").write_text(
        json.dumps(registros, ensure_ascii=False), encoding="utf-8"
    )
    cartao = {
        "recorte": {
            "anos": args.anos,
            "tipo": args.tipo,
            "maximo_por_ano": args.maximo_por_ano,
        },
        "baixado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "minutos_de_coleta": round(minutos, 1),
        "fonte": "https://dadosabertos.camara.leg.br/api/v2",
        "taxonomia": {
            "origem": "lista oficial em /referencias/proposicoes/codTema",
            "n_temas": len(temas),
            "temas": sorted(temas.values()),
        },
        "coleta": {
            "proposicoes": len(registros),
            "por_ano": por_ano,
            "com_ao_menos_um_tema": com_tema,
            "sem_tema": len(registros) - com_tema,
            "falhas": falhas[:50],
            "n_falhas": len(falhas),
            "coleta_completa": not falhas,
        },
    }
    (saida / "cartao_do_dado.json").write_text(
        json.dumps(cartao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nartefatos em {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
