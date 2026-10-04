"""Testes dos candidatos e da preparacao do texto.

O teste mais importante do arquivo e o do vazamento de anotacao: ele fixa que
`montar_texto` so inclui as keywords quando alguem pede explicitamente. Se esse
padrao virar `True` um dia, todo numero do projeto sobe e nada falha.
"""

from __future__ import annotations

import numpy as np
import pytest

from verbete.aplicacao.classificador import (
    PorPalavraChave,
    SempreOsFrequentes,
    TfidfLinear,
    matriz_de_rotulos,
    montar_texto,
    normalizar,
)

TEMAS = ["Saúde", "Educação", "Meio Ambiente e Desenvolvimento Sustentável"]


def _registro(ementa: str, temas: list[str], keywords: str = "") -> dict:
    return {
        "ementa": ementa,
        "ementa_detalhada": "",
        "keywords": keywords,
        "temas": temas,
    }


# ------------------------------------------------------------- normalizacao


def test_normalizar_tira_acento_e_pontuacao():
    assert normalizar("Saúde, Educação e Meio-Ambiente!") == (
        "saude educacao e meio ambiente"
    )


def test_normalizar_colapsa_espaco_e_aceita_vazio():
    assert normalizar("  a   b  ") == "a b"
    assert normalizar("") == ""
    assert normalizar(None) == ""


def test_normalizar_remove_numero():
    """Numero de lei nao distingue tema e inflaria o vocabulario."""
    assert "8213" not in normalizar("Altera a Lei nº 8.213, de 1991")


# ------------------------------------------- o vazamento de anotacao


def test_keywords_ficam_de_fora_por_padrao():
    """A defesa principal do projeto, num teste de uma linha.

    As keywords sao preenchidas pela mesma indexacao humana que atribui o tema.
    Se o padrao virasse True, todo numero subiria e nada falharia.
    """
    r = _registro("texto da ementa", ["Saúde"], keywords="saude, hospital, sus")
    assert "hospital" not in montar_texto(r, usar_keywords=False)
    assert "hospital" in montar_texto(r, usar_keywords=True)


def test_ementa_detalhada_entra_por_padrao_e_pode_sair():
    r = {
        "ementa": "curta",
        "ementa_detalhada": "detalhada",
        "keywords": "",
        "temas": [],
    }
    assert "detalhada" in montar_texto(r, usar_keywords=False)
    assert "detalhada" not in montar_texto(
        r, usar_keywords=False, usar_ementa_detalhada=False
    )


def test_campo_ausente_nao_quebra():
    assert montar_texto({"temas": []}, usar_keywords=True) == ""


# --------------------------------------------------------- matriz de rotulos


def test_matriz_marca_os_temas_certos():
    regs = [_registro("a", ["Saúde"]), _registro("b", ["Saúde", "Educação"])]
    y = matriz_de_rotulos(regs, TEMAS)
    assert y.tolist() == [[True, False, False], [True, True, False]]


def test_tema_fora_da_taxonomia_e_ignorado_sem_quebrar():
    """A taxonomia oficial manda. Tema estranho nao vira coluna nova."""
    y = matriz_de_rotulos([_registro("a", ["Inventado"])], TEMAS)
    assert not y.any()


def test_proposicao_sem_tema_vira_linha_zerada():
    y = matriz_de_rotulos([_registro("a", [])], TEMAS)
    assert y.shape == (1, 3)
    assert not y.any()


# ----------------------------------------------------- sempre os frequentes


def test_sempre_os_frequentes_escolhe_os_mais_comuns():
    y = np.zeros((10, 3), dtype=bool)
    y[:8, 0] = True
    y[:3, 1] = True
    m = SempreOsFrequentes(k=1).ajustar(["x"] * 10, y)
    p = m.prever(["qualquer coisa"])
    assert p.tolist() == [[True, False, False]]


def test_sempre_os_frequentes_nao_le_o_texto():
    y = np.zeros((6, 2), dtype=bool)
    y[:, 0] = True
    m = SempreOsFrequentes(k=1).ajustar(["a"] * 6, y)
    a = m.prever(["saude hospital"])
    b = m.prever(["educacao escola"])
    assert a.tolist() == b.tolist()


def test_k_padrao_vem_da_media_de_temas_por_linha():
    """Prever mais temas que a media inflaria a revocacao de graca."""
    y = np.zeros((10, 4), dtype=bool)
    y[:, 0] = True
    y[:, 1] = True
    m = SempreOsFrequentes().ajustar(["x"] * 10, y)
    assert m.prever(["x"]).sum() == 2


# ------------------------------------------------------- por palavra chave


def test_palavra_chave_dispara_quando_o_termo_aparece():
    m = PorPalavraChave(temas=TEMAS).ajustar([], np.zeros((0, 3), dtype=bool))
    p = m.prever([normalizar("Dispõe sobre a saúde pública")])
    assert p[0, 0]
    assert not p[0, 1]


def test_palavra_chave_ignora_palavra_vazia_do_nome_do_tema():
    """'Meio Ambiente e Desenvolvimento Sustentável' nao pode disparar por 'e'."""
    m = PorPalavraChave(temas=TEMAS).ajustar([], np.zeros((0, 3), dtype=bool))
    p = m.prever([normalizar("Dispõe sobre a pesca e a caça")])
    assert not p[0, 2]


# ------------------------------------------------------------ tfidf linear


def _corpo_sintetico(n: int = 400, semente: int = 0):
    """Corpus em que cada tema tem vocabulario proprio. Resposta conhecida."""
    rng = np.random.default_rng(semente)
    vocabulario = {
        0: "hospital medico paciente leito enfermagem",
        1: "escola aluno professor matricula ensino",
        2: "floresta residuo poluicao reciclagem desmatamento",
    }
    textos, y = [], np.zeros((n, 3), dtype=bool)
    for i in range(n):
        j = int(rng.integers(0, 3))
        y[i, j] = True
        palavras = vocabulario[j].split()
        rng.shuffle(palavras)
        textos.append(" ".join(palavras[: rng.integers(3, 6)]))
    return textos, y


def test_tfidf_aprende_vocabulario_que_separa():
    textos, y = _corpo_sintetico()
    m = TfidfLinear(min_df=1).ajustar(textos, y)
    from verbete.dominio.avaliacao import micro

    assert micro(y, m.prever(textos))["f1"] > 0.95


def test_tfidf_nao_aprende_relacao_que_nao_existe():
    """Rotulo sorteado sobre vocabulario comum: nao pode haver sinal.

    Com rotulo independente do texto, um F1 alto no proprio treino significaria
    sobreajuste grosseiro, e qualquer resultado do projeto ficaria suspeito.
    """
    rng = np.random.default_rng(7)
    palavras = [
        "alfa", "bravo", "charlie", "delta", "eco", "foxtrot", "golfo", "hotel",
    ]
    textos = [" ".join(rng.choice(palavras, 6)) for _ in range(300)]
    y = rng.random((300, 3)) < 0.3
    m = TfidfLinear(min_df=1).ajustar(textos, y)
    from verbete.dominio.avaliacao import micro

    assert micro(y, m.prever(textos))["f1"] < 0.60


def test_vocabulario_vazio_da_erro_que_diz_o_que_fazer():
    """A mensagem da biblioteca nao ajuda; a do dominio diz qual botao mexer."""
    from verbete.aplicacao.classificador import VocabularioVazio

    with pytest.raises(VocabularioVazio, match="min_df"):
        TfidfLinear(min_df=50).ajustar(["um texto curto"] * 3, np.zeros((3, 2), dtype=bool))


def test_tfidf_exige_ajuste_antes_de_prever():
    with pytest.raises(RuntimeError, match="ajustar antes"):
        TfidfLinear().escores(["x"])


def test_rotulo_com_um_caso_so_nao_vira_modelo():
    """Treinar com um exemplo produz um modelo que dispara em qualquer coisa."""
    textos, y = _corpo_sintetico(n=60)
    y[:, 2] = False
    y[0, 2] = True
    m = TfidfLinear(min_df=1).ajustar(textos, y)
    assert m.modelos[2] is None
    assert m.escores(textos)[:, 2].sum() == 0.0


def test_termos_que_pesam_saem_do_proprio_modelo():
    """A explicacao e o coeficiente, nao uma aproximacao posterior."""
    textos, y = _corpo_sintetico()
    m = TfidfLinear(min_df=1).ajustar(textos, y)
    d = m.termos_que_pesam(0, quantos=5)
    a_favor = [x["termo"] for x in d["a_favor"]]
    assert any(t in a_favor for t in ("hospital", "medico", "paciente", "leito"))
    assert d["a_favor"][0]["peso"] > d["contra"][0]["peso"]


def test_termos_de_rotulo_sem_modelo_vem_vazio():
    textos, y = _corpo_sintetico(n=60)
    y[:, 2] = False
    m = TfidfLinear(min_df=1).ajustar(textos, y)
    assert m.termos_que_pesam(2) == {"a_favor": [], "contra": []}


def test_limiar_por_rotulo_muda_a_previsao():
    textos, y = _corpo_sintetico()
    m = TfidfLinear(min_df=1).ajustar(textos, y)
    baixo = m.prever(textos, limiares=0.1).sum()
    alto = m.prever(textos, limiares=0.9).sum()
    assert baixo > alto
