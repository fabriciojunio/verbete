"""Metricas para classificacao multirrotulo, e a decisao de nao responder.

**Por que nao basta acuracia.** Com 32 temas e cerca de 1,2 tema por proposicao,
um modelo que responde "nenhum tema" acerta a maioria das 32 celulas de cada
linha. Acuracia por celula fica acima de 95% e o modelo e inutil. As metricas
que valem sao as que olham o rotulo positivo:

    micro-F1    agrega acerto e erro de todos os rotulos juntos; e dominada
                pelos temas frequentes
    macro-F1    media dos F1 por rotulo; da o mesmo peso a um tema com 900
                casos e a um com 12, e por isso e a metrica honesta quando o
                objetivo e cobrir a taxonomia inteira

As duas aparecem sempre juntas. A distancia entre elas **e** o resultado: ela
mede o quanto o modelo esta funcionando so nos temas comuns.

**A decisao de nao responder.** Num produto de monitoramento regulatorio, errar
o tema custa mais do que nao opinar: quem recebe a classificacao age sobre ela.
Por isso o modulo calcula a curva de cobertura contra acerto, que responde
"quanto de precisao eu ganho ao deixar o modelo calar nos casos em que ele esta
inseguro, e quanta vazao eu perco".
"""

from __future__ import annotations

import numpy as np


def _binario(y) -> np.ndarray:
    return np.asarray(y).astype(bool)


def contagens(y: np.ndarray, p: np.ndarray) -> tuple[int, int, int]:
    y, p = _binario(y), _binario(p)
    vp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    return vp, fp, fn


def precisao_revocacao_f1(y: np.ndarray, p: np.ndarray) -> dict:
    vp, fp, fn = contagens(y, p)
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = (
        2 * precisao * revocacao / (precisao + revocacao)
        if precisao + revocacao
        else 0.0
    )
    return {
        "precisao": precisao,
        "revocacao": revocacao,
        "f1": f1,
        "vp": vp,
        "fp": fp,
        "fn": fn,
    }


def micro(y: np.ndarray, p: np.ndarray) -> dict:
    """Agrega todas as celulas antes de calcular. Dominada pelo tema frequente."""
    return precisao_revocacao_f1(np.asarray(y).ravel(), np.asarray(p).ravel())


def macro(y: np.ndarray, p: np.ndarray) -> dict:
    """Media por rotulo. Um tema raro pesa o mesmo que um comum.

    Rotulo sem nenhum caso positivo no conjunto fica **de fora** da media: F1 de
    um rotulo que nao aparece nao e zero, e indefinido, e contar como zero
    puniria o modelo por um rotulo que ele nao teve chance de acertar.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    por_rotulo = []
    for j in range(y.shape[1]):
        if not y[:, j].any():
            continue
        por_rotulo.append(precisao_revocacao_f1(y[:, j], p[:, j]))
    if not por_rotulo:
        return {"precisao": 0.0, "revocacao": 0.0, "f1": 0.0, "rotulos": 0}
    return {
        "precisao": float(np.mean([x["precisao"] for x in por_rotulo])),
        "revocacao": float(np.mean([x["revocacao"] for x in por_rotulo])),
        "f1": float(np.mean([x["f1"] for x in por_rotulo])),
        "rotulos": len(por_rotulo),
    }


def por_rotulo(
    y: np.ndarray, p: np.ndarray, nomes: list[str]
) -> list[dict]:
    y = np.asarray(y)
    p = np.asarray(p)
    saida = []
    for j, nome in enumerate(nomes):
        m = precisao_revocacao_f1(y[:, j], p[:, j])
        saida.append(
            {
                "rotulo": nome,
                "apoio": int(y[:, j].sum()),
                "previstos": int(p[:, j].sum()),
                **m,
            }
        )
    return sorted(saida, key=lambda d: -d["apoio"])


def acerto_exato(y: np.ndarray, p: np.ndarray) -> float:
    """Fracao de linhas em que o conjunto de temas saiu exatamente igual.

    E a metrica mais dura das que fazem sentido, e a mais proxima do que o
    usuario percebe: ou a classificacao da proposicao esta certa, ou nao esta.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    return float((y.astype(bool) == p.astype(bool)).all(axis=1).mean())


def ao_menos_um_certo(y: np.ndarray, p: np.ndarray) -> float:
    """Fracao de linhas em que ao menos um tema previsto esta entre os certos.

    Util porque num produto de monitoramento acertar um dos dois temas ja leva o
    documento para a mesa certa.
    """
    y = _binario(y)
    p = _binario(p)
    tem_previsao = p.any(axis=1)
    if not tem_previsao.any():
        return 0.0
    return float((y & p).any(axis=1)[tem_previsao].mean())


def avaliar(y: np.ndarray, p: np.ndarray, nomes: list[str] | None = None) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    d = {
        "n": int(y.shape[0]),
        "rotulos": int(y.shape[1]),
        "micro_f1": micro(y, p)["f1"],
        "macro_f1": macro(y, p)["f1"],
        "micro_precisao": micro(y, p)["precisao"],
        "micro_revocacao": micro(y, p)["revocacao"],
        "acerto_exato": acerto_exato(y, p),
        "ao_menos_um_certo": ao_menos_um_certo(y, p),
        "temas_previstos_por_linha": float(np.asarray(p).sum(axis=1).mean()),
        "temas_certos_por_linha": float(np.asarray(y).sum(axis=1).mean()),
    }
    d["distancia_micro_macro"] = d["micro_f1"] - d["macro_f1"]
    if nomes is not None:
        d["por_rotulo"] = por_rotulo(y, p, nomes)
    return d


# ------------------------------------------------------- decidir nao responder


def curva_de_abstencao(
    y: np.ndarray,
    escores: np.ndarray,
    limiar_de_decisao: float,
    cortes_de_confianca: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    ),
) -> list[dict]:
    """Quanto se ganha de acerto ao deixar o modelo calar quando esta inseguro.

    A confianca de uma linha e o **maior escore** entre os rotulos. Quando ela
    fica abaixo do corte, a linha vai para revisao humana e nao conta como
    acerto nem como erro: conta como cobertura perdida.

    Esta e a tabela que decide se o produto serve: um modelo com micro-F1 de
    0,60 em 100% dos casos pode ser pior, para quem usa, do que o mesmo modelo
    com 0,80 em 70% dos casos e revisao humana nos 30% restantes.
    """
    y = np.asarray(y)
    escores = np.asarray(escores, dtype=float)
    confianca = escores.max(axis=1)
    previsto_total = escores >= limiar_de_decisao
    linhas = []
    for corte in cortes_de_confianca:
        atende = confianca >= corte
        cobertura = float(atende.mean())
        if not atende.any():
            linhas.append(
                {
                    "corte_de_confianca": corte,
                    "cobertura": 0.0,
                    "n_respondidos": 0,
                    "micro_f1": float("nan"),
                    "macro_f1": float("nan"),
                    "acerto_exato": float("nan"),
                }
            )
            continue
        yy, pp = y[atende], previsto_total[atende]
        linhas.append(
            {
                "corte_de_confianca": corte,
                "cobertura": cobertura,
                "n_respondidos": int(atende.sum()),
                "micro_f1": micro(yy, pp)["f1"],
                "macro_f1": macro(yy, pp)["f1"],
                "acerto_exato": acerto_exato(yy, pp),
            }
        )
    return linhas


def melhor_limiar_por_rotulo(
    y: np.ndarray,
    escores: np.ndarray,
    candidatos: np.ndarray | None = None,
) -> np.ndarray:
    """Um limiar por rotulo, escolhido para maximizar o F1 daquele rotulo.

    Limiar unico para 32 temas de frequencias muito diferentes e uma escolha
    ruim disfarcada de simplicidade: o corte que serve para um tema com 900
    casos nunca dispara para um com 12. O ajuste e feito **na validacao**, nunca
    no teste, e o vetor de limiares entra no cartao do modelo, porque ele faz
    parte do modelo e nao da avaliacao.
    """
    y = np.asarray(y)
    escores = np.asarray(escores, dtype=float)
    if candidatos is None:
        candidatos = np.linspace(0.05, 0.95, 19)
    limiares = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if not y[:, j].any():
            continue
        melhor, melhor_f1 = 0.5, -1.0
        for c in candidatos:
            f1 = precisao_revocacao_f1(y[:, j], escores[:, j] >= c)["f1"]
            if f1 > melhor_f1:
                melhor, melhor_f1 = float(c), f1
        limiares[j] = melhor
    return limiares
