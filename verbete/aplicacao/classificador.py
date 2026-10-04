"""Os candidatos a classificador, e o que cada um existe para provar.

A ordem importa, e ela e a mesma de sempre: **primeiro o que alguem faria sem
modelo**, depois o modelo, e a conta de quanto o segundo ganha do primeiro.

    sempre_os_frequentes   prevê os k temas mais comuns para toda proposicao.
                           Nao le o texto. E o piso, e e surpreendente quantas
                           vezes ele ganha de modelo mal avaliado
    por_palavra_chave      regra de dicionario: termo do nome do tema aparece na
                           ementa, entao o tema entra. E o que uma pessoa monta
                           numa tarde, e e a alternativa honesta ao modelo
    tfidf_linear           TF-IDF com n-gramas de palavra, mais uma logistica
                           por tema. Linear de proposito: o coeficiente **e** a
                           explicacao, e num produto regulado explicar vale mais
                           que meio ponto de F1

**Por que nao um transformador.** Ele provavelmente ganharia. Nao esta aqui
porque o projeto nao teria como medir o ganho de forma honesta sem GPU e sem
tempo de ajuste, e um numero de um modelo mal ajustado seria pior que nenhum
numero. O limite esta declarado no README em vez de disfarcado.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

# As palavras que aparecem em quase toda ementa e nao distinguem tema nenhum.
# A lista e curta de proposito: lista grande de palavras vazias costuma remover
# termo que importa ("nao", "sem") e o TF-IDF ja cuida do resto pelo peso.
VAZIAS = frozenset(
    ["a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sobre", "ao", "aos", "à", "às", "e", "ou", "que", "se", "como", "ser", "sao", "e'", "sua", "seu", "suas", "seus", "este", "esta", "estes", "estas", "esse", "essa", "isso", "aquele", "aquela", "pelo", "pela", "pelos", "pelas", "lei", "art", "artigo", "n", "numero", "dispoe", "dispõe", "altera", "acrescenta", "da", "da"]
)


class VocabularioVazio(ValueError):
    """Nenhum termo sobreviveu ao vetorizador.

    Tipo proprio porque a causa e sempre de configuracao, nunca de dado
    corrompido, e quem le o erro precisa saber qual botao mexer.
    """


def normalizar(texto: str) -> str:
    """Minusculas, sem acento, so letras e espaco.

    Tirar acento e decisao com custo: "publico" e "público" viram a mesma
    coisa, o que ajuda, mas tambem colide palavras diferentes. Vale neste caso
    porque a ementa e escrita por muitas mãos e a acentuacao varia, e um termo
    escrito dos dois jeitos viraria dois termos com metade da frequencia cada.
    """
    if not texto:
        return ""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def montar_texto(
    registro: dict, usar_keywords: bool, usar_ementa_detalhada: bool = True
) -> str:
    """O texto que entra no modelo, com a origem de cada pedaco declarada.

    `keywords` fica atras de uma chave porque e **vazamento de anotacao**: ela e
    preenchida pela mesma indexacao que atribui o tema. O projeto mede com e sem,
    e a diferenca entre os dois numeros e exatamente o tamanho do vazamento.
    """
    partes = [registro.get("ementa") or ""]
    if usar_ementa_detalhada:
        partes.append(registro.get("ementa_detalhada") or "")
    if usar_keywords:
        partes.append(registro.get("keywords") or "")
    return normalizar(" ".join(partes))


def matriz_de_rotulos(
    registros: list[dict], temas: list[str]
) -> np.ndarray:
    indice = {t: j for j, t in enumerate(temas)}
    y = np.zeros((len(registros), len(temas)), dtype=bool)
    for i, r in enumerate(registros):
        for t in r.get("temas") or []:
            j = indice.get(t)
            if j is not None:
                y[i, j] = True
    return y


# --------------------------------------------------------------- os candidatos


@dataclass
class SempreOsFrequentes:
    """Prevê os k temas mais comuns do treino, sem olhar o texto.

    `k` sai da media de temas por proposicao no treino, arredondada: prever mais
    temas do que a media infla a revocacao de graca, e o piso tem de ser um piso
    honesto, nao um espantalho.
    """

    k: int | None = None
    frequentes: np.ndarray | None = field(default=None, init=False)
    n_rotulos: int = field(default=0, init=False)

    def ajustar(self, textos: list[str], y: np.ndarray) -> SempreOsFrequentes:
        y = np.asarray(y)
        self.n_rotulos = y.shape[1]
        k = self.k if self.k is not None else max(1, int(round(y.sum(axis=1).mean())))
        self.frequentes = np.argsort(-y.sum(axis=0))[:k]
        return self

    def escores(self, textos: list[str]) -> np.ndarray:
        s = np.zeros((len(textos), self.n_rotulos))
        s[:, self.frequentes] = 1.0
        return s

    def prever(self, textos: list[str]) -> np.ndarray:
        return self.escores(textos) >= 0.5


@dataclass
class PorPalavraChave:
    """Regra de dicionario: o nome do tema aparece na ementa, o tema entra.

    E a alternativa que nao precisa de aprendizado nenhum, e ela existe para
    responder a pergunta que ninguem faz antes de treinar: sera que basta
    procurar a palavra?
    """

    temas: list[str] = field(default_factory=list)
    termos: list[list[str]] = field(default_factory=list, init=False)

    def ajustar(self, textos: list[str], y: np.ndarray) -> PorPalavraChave:
        self.termos = []
        for t in self.temas:
            palavras = [
                p for p in normalizar(t).split() if p not in VAZIAS and len(p) > 3
            ]
            self.termos.append(palavras)
        return self

    def escores(self, textos: list[str]) -> np.ndarray:
        s = np.zeros((len(textos), len(self.temas)))
        for i, texto in enumerate(textos):
            for j, palavras in enumerate(self.termos):
                if palavras and any(p in texto for p in palavras):
                    s[i, j] = 1.0
        return s

    def prever(self, textos: list[str]) -> np.ndarray:
        return self.escores(textos) >= 0.5


@dataclass
class TfidfLinear:
    """TF-IDF mais uma logistica por tema, uma contra o resto.

    Linear de proposito. Num produto de monitoramento regulatorio alguem vai
    perguntar por que a proposicao foi para a mesa de tributario, e com modelo
    linear a resposta e a lista de termos que mais pesaram, extraida do proprio
    modelo e nao de uma aproximacao posterior.

    `class_weight="balanced"` nao e detalhe: com 32 temas e distribuicao muito
    desigual, sem ele os temas raros nunca disparam e o macro-F1 desaba enquanto
    o micro-F1 continua bonito.
    """

    min_df: int = 3
    max_features: int = 60_000
    ngramas: tuple[int, int] = (1, 2)
    regularizacao: float = 1.0
    vetorizador: object | None = field(default=None, init=False)
    modelos: list = field(default_factory=list, init=False)
    n_rotulos: int = field(default=0, init=False)

    def ajustar(self, textos: list[str], y: np.ndarray) -> TfidfLinear:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y)
        self.n_rotulos = y.shape[1]
        self.vetorizador = TfidfVectorizer(
            min_df=self.min_df,
            max_features=self.max_features,
            ngram_range=self.ngramas,
            stop_words=sorted(VAZIAS),
            sublinear_tf=True,
        )
        try:
            x = self.vetorizador.fit_transform(textos)
        except ValueError as exc:
            # a mensagem da biblioteca ("empty vocabulary") nao diz o que fazer.
            # As causas reais sao duas e as duas tem conserto: min_df alto demais
            # para o tamanho do corpus, ou texto que virou vazio na normalizacao
            raise VocabularioVazio(
                f"nenhum termo sobreviveu ao vetorizador em {len(textos)} "
                f"documentos com min_df={self.min_df}. Ou o corpus e pequeno "
                f"demais para esse min_df, ou a normalizacao esvaziou o texto"
            ) from exc
        self.modelos = []
        for j in range(self.n_rotulos):
            alvo = y[:, j].astype(int)
            if alvo.sum() < 2 or alvo.sum() == len(alvo):
                # rotulo sem casos suficientes: guarda None e devolve escore
                # zero. Treinar com um exemplo so produziria um modelo que
                # dispara em qualquer coisa parecida com aquele exemplo
                self.modelos.append(None)
                continue
            self.modelos.append(
                LogisticRegression(
                    C=self.regularizacao,
                    max_iter=2000,
                    class_weight="balanced",
                ).fit(x, alvo)
            )
        return self

    def escores(self, textos: list[str]) -> np.ndarray:
        if self.vetorizador is None:
            raise RuntimeError("ajustar antes de prever")
        x = self.vetorizador.transform(textos)
        s = np.zeros((len(textos), self.n_rotulos))
        for j, m in enumerate(self.modelos):
            if m is not None:
                s[:, j] = m.predict_proba(x)[:, 1]
        return s

    def prever(
        self, textos: list[str], limiares: np.ndarray | float = 0.5
    ) -> np.ndarray:
        return self.escores(textos) >= limiares

    def termos_que_pesam(self, rotulo: int, quantos: int = 12) -> dict:
        """Os termos de maior e menor peso para um tema. A explicacao do modelo.

        Nao e aproximacao posterior nem atribuicao por perturbacao: sao os
        coeficientes do proprio classificador que decide.
        """
        m = self.modelos[rotulo] if rotulo < len(self.modelos) else None
        if m is None or self.vetorizador is None:
            return {"a_favor": [], "contra": []}
        nomes = np.asarray(self.vetorizador.get_feature_names_out())
        coef = m.coef_[0]
        ordem = np.argsort(coef)
        return {
            "a_favor": [
                {"termo": str(nomes[i]), "peso": float(coef[i])}
                for i in ordem[::-1][:quantos]
            ],
            "contra": [
                {"termo": str(nomes[i]), "peso": float(coef[i])}
                for i in ordem[:quantos]
            ],
        }
