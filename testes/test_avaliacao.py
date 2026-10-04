"""Testes das metricas multirrotulo, contra respostas conhecidas.

O caso que mais importa aqui e o do modelo que nao prevê nada. Com 32 temas e
pouco mais de um tema por proposicao, ele acerta a esmagadora maioria das
celulas, e qualquer metrica que nao distinga isso de um modelo bom torna o
projeto inutil sem dar erro.
"""

from __future__ import annotations

import numpy as np
import pytest

from verbete.dominio.avaliacao import (
    acerto_exato,
    ao_menos_um_certo,
    avaliar,
    curva_de_abstencao,
    macro,
    melhor_limiar_por_rotulo,
    micro,
    por_rotulo,
    precisao_revocacao_f1,
)

# ------------------------------------------------------- valores fechados


def test_f1_perfeito():
    y = np.array([[1, 0], [0, 1]], dtype=bool)
    m = precisao_revocacao_f1(y, y)
    assert m["f1"] == pytest.approx(1.0)
    assert m["fp"] == 0 and m["fn"] == 0


def test_f1_tem_valor_fechado():
    """2 verdadeiros positivos, 1 falso positivo, 1 falso negativo.

    Precisao 2/3, revocacao 2/3, F1 2/3.
    """
    y = np.array([1, 1, 0, 1])
    p = np.array([1, 1, 1, 0])
    m = precisao_revocacao_f1(y, p)
    assert m["precisao"] == pytest.approx(2 / 3)
    assert m["revocacao"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_modelo_que_nao_preve_nada_tem_f1_zero():
    """O caso que o projeto mais precisa distinguir.

    Ele acerta quase toda celula da matriz, e acuracia por celula o premiaria.
    F1 o reprova, que e o comportamento certo.
    """
    y = np.zeros((100, 32), dtype=bool)
    y[np.arange(100), np.random.default_rng(0).integers(0, 32, 100)] = True
    p = np.zeros_like(y)
    assert (y == p).mean() > 0.95, "a acuracia por celula premiaria este modelo"
    assert micro(y, p)["f1"] == 0.0
    assert macro(y, p)["f1"] == 0.0


def test_modelo_que_preve_tudo_tem_revocacao_um_e_precisao_baixa():
    y = np.zeros((50, 32), dtype=bool)
    y[:, 0] = True
    p = np.ones_like(y)
    m = micro(y, p)
    assert m["revocacao"] == pytest.approx(1.0)
    assert m["precisao"] == pytest.approx(1 / 32)


# ----------------------------------------------------------- micro e macro


def test_macro_e_micro_coincidem_com_rotulos_equilibrados():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
    p = y.copy()
    assert micro(y, p)["f1"] == pytest.approx(macro(y, p)["f1"])


def test_macro_cai_quando_o_rotulo_raro_e_ignorado():
    """O resultado que a distancia entre micro e macro mede.

    Um tema com 98 casos e outro com 2. O modelo acerta o comum e ignora o raro:
    o micro-F1 continua alto e o macro cai pela metade.
    """
    y = np.zeros((100, 2), dtype=bool)
    y[:98, 0] = True
    y[98:, 1] = True
    p = np.zeros_like(y)
    p[:98, 0] = True
    assert micro(y, p)["f1"] > 0.98
    assert macro(y, p)["f1"] == pytest.approx(0.5)


def test_macro_ignora_rotulo_sem_nenhum_caso():
    """F1 de um rotulo que nao aparece e indefinido, nao zero.

    Conta-lo como zero puniria o modelo por um rotulo que ele nao teve chance de
    acertar, e com 32 temas e um recorte pequeno isso acontece sempre.
    """
    y = np.zeros((10, 3), dtype=bool)
    y[:, 0] = True
    p = y.copy()
    m = macro(y, p)
    assert m["rotulos"] == 1
    assert m["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------- acerto por linha


def test_acerto_exato_exige_o_conjunto_inteiro():
    y = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
    p = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
    assert acerto_exato(y, p) == pytest.approx(0.5)


def test_ao_menos_um_certo_e_mais_brando_que_o_exato():
    y = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
    p = np.array([[1, 0, 0], [1, 1, 0]], dtype=bool)
    assert acerto_exato(y, p) == pytest.approx(0.0)
    assert ao_menos_um_certo(y, p) == pytest.approx(1.0)


def test_ao_menos_um_certo_ignora_linha_sem_previsao():
    """Linha em que o modelo nao opinou nao conta como acerto nem como erro.

    Contar como erro misturaria duas coisas: errar e nao responder. A curva de
    abstencao existe justamente para separa-las.
    """
    y = np.array([[1, 0], [1, 0]], dtype=bool)
    p = np.array([[1, 0], [0, 0]], dtype=bool)
    assert ao_menos_um_certo(y, p) == pytest.approx(1.0)


def test_por_rotulo_sai_ordenado_por_apoio():
    y = np.zeros((10, 3), dtype=bool)
    y[:8, 0] = True
    y[:5, 1] = True
    y[:1, 2] = True
    linhas = por_rotulo(y, y, ["a", "b", "c"])
    assert [x["rotulo"] for x in linhas] == ["a", "b", "c"]
    assert [x["apoio"] for x in linhas] == [8, 5, 1]


def test_avaliar_reune_tudo_e_calcula_a_distancia():
    y = np.zeros((100, 2), dtype=bool)
    y[:98, 0] = True
    y[98:, 1] = True
    p = np.zeros_like(y)
    p[:98, 0] = True
    d = avaliar(y, p, nomes=["comum", "raro"])
    assert d["distancia_micro_macro"] == pytest.approx(
        d["micro_f1"] - d["macro_f1"]
    )
    assert d["distancia_micro_macro"] > 0.4
    assert len(d["por_rotulo"]) == 2


# ----------------------------------------------------------- abstencao


def test_abstencao_com_corte_zero_responde_tudo():
    rng = np.random.default_rng(1)
    y = rng.random((200, 5)) < 0.2
    s = rng.random((200, 5))
    linhas = curva_de_abstencao(y, s, limiar_de_decisao=0.5)
    assert linhas[0]["corte_de_confianca"] == 0.0
    assert linhas[0]["cobertura"] == pytest.approx(1.0)
    assert linhas[0]["n_respondidos"] == 200


def test_abstencao_reduz_cobertura_de_forma_monotona():
    rng = np.random.default_rng(2)
    y = rng.random((300, 6)) < 0.2
    s = rng.random((300, 6))
    linhas = curva_de_abstencao(y, s, limiar_de_decisao=0.5)
    coberturas = [x["cobertura"] for x in linhas]
    assert coberturas == sorted(coberturas, reverse=True)


def test_abstencao_melhora_o_acerto_quando_o_escore_informa():
    """O resultado que justifica a abstencao existir.

    Aqui o escore e informativo: quando ele e alto, o rotulo esta certo. Calar
    nos casos de escore baixo tem de aumentar o acerto entre os respondidos.
    """
    rng = np.random.default_rng(3)
    n = 400
    y = np.zeros((n, 3), dtype=bool)
    s = np.zeros((n, 3))
    for i in range(n):
        j = rng.integers(0, 3)
        y[i, j] = True
        confiante = i < n // 2
        s[i, j] = rng.uniform(0.8, 0.99) if confiante else rng.uniform(0.4, 0.6)
        outros = [k for k in range(3) if k != j]
        s[i, outros] = rng.uniform(0.0, 0.3, size=2) if confiante else rng.uniform(0.3, 0.55, size=2)
    linhas = curva_de_abstencao(y, s, limiar_de_decisao=0.5)
    sem_corte = linhas[0]["micro_f1"]
    com_corte = [x for x in linhas if x["corte_de_confianca"] == 0.7][0]
    assert com_corte["micro_f1"] > sem_corte
    assert com_corte["cobertura"] < 1.0


def test_abstencao_com_corte_alto_demais_nao_quebra():
    y = np.zeros((10, 2), dtype=bool)
    s = np.full((10, 2), 0.1)
    linhas = curva_de_abstencao(y, s, 0.5, cortes_de_confianca=(0.9,))
    assert linhas[0]["n_respondidos"] == 0
    assert np.isnan(linhas[0]["micro_f1"])


# ------------------------------------------------------- limiar por rotulo


def test_limiar_por_rotulo_acha_o_corte_que_separa():
    y = np.array([[1], [1], [0], [0]], dtype=bool)
    s = np.array([[0.9], [0.8], [0.2], [0.1]])
    limiar = melhor_limiar_por_rotulo(y, s)
    assert 0.2 < limiar[0] <= 0.8


def test_limiar_por_rotulo_difere_entre_rotulos_de_frequencia_diferente():
    """E o motivo de existir: um corte so para 32 temas e simplicidade falsa."""
    rng = np.random.default_rng(4)
    n = 600
    y = np.zeros((n, 2), dtype=bool)
    y[: n // 2, 0] = True      # tema comum
    y[: n // 50, 1] = True     # tema raro
    s = np.zeros((n, 2))
    s[:, 0] = np.where(y[:, 0], rng.uniform(0.5, 0.9, n), rng.uniform(0.1, 0.5, n))
    s[:, 1] = np.where(y[:, 1], rng.uniform(0.2, 0.4, n), rng.uniform(0.0, 0.2, n))
    limiares = melhor_limiar_por_rotulo(y, s)
    assert limiares[0] != limiares[1]
    assert limiares[1] < limiares[0], "o tema raro precisa de um corte mais baixo"


def test_limiar_de_rotulo_sem_caso_fica_no_padrao():
    y = np.zeros((10, 2), dtype=bool)
    y[:5, 0] = True
    limiares = melhor_limiar_por_rotulo(y, np.random.default_rng(5).random((10, 2)))
    assert limiares[1] == pytest.approx(0.5)
