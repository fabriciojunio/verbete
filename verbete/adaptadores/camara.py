"""Leitor da API de dados abertos da Camara dos Deputados.

A API e publica, nao pede chave e responde rapido, o que e o oposto do que
costuma acontecer com dado publico brasileiro. Mesmo assim o cliente aqui tem
retentativa e pausa entre chamadas: a coleta faz milhares de requisicoes e
ninguem deve derrubar um servico publico por nao ter pensado nisso.

**O que importa para o projeto.** Cada proposicao traz:

    ementa            o resumo curto, que e o texto principal
    ementaDetalhada   um resumo mais longo, nem sempre preenchido
    keywords          palavras-chave atribuidas POR PESSOA na indexacao
    temas             de 1 a N temas, de uma lista oficial de 32

Os `temas` sao o rotulo, e as `keywords` sao a armadilha: elas tambem sao
atribuidas na indexacao, pela mesma pessoa e no mesmo momento em que o tema e
atribuido. Usa-las como variavel e usar parte do trabalho humano que produziu o
rotulo, e o modelo parece melhor do que vai ser em producao, onde a proposicao
chega sem indexacao nenhuma. O projeto mede os dois casos e declara a diferenca.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE = "https://dadosabertos.camara.leg.br/api/v2"


class FalhaDaCamara(RuntimeError):
    """A API nao respondeu depois de todas as tentativas."""


@dataclass
class ClienteCamara:
    pausa_entre_chamadas: float = 0.12
    tentativas: int = 4
    espera_inicial: float = 2.0
    tempo_limite: int = 60

    def _get(self, caminho: str) -> dict:
        url = f"{BASE}{caminho}"
        ultimo: Exception | None = None
        for tentativa in range(self.tentativas):
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.tempo_limite) as r:
                    return json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                ultimo = exc
                if tentativa + 1 < self.tentativas:
                    time.sleep(self.espera_inicial * (2**tentativa))
        raise FalhaDaCamara(f"{url} falhou apos {self.tentativas} tentativas: {ultimo}")

    def temas_oficiais(self) -> dict[int, str]:
        """A lista oficial de temas. E a taxonomia, e nao foi inventada aqui."""
        d = self._get("/referencias/proposicoes/codTema")
        return {int(t["cod"]): t["nome"] for t in d["dados"]}

    def proposicoes(
        self,
        ano: int,
        sigla_tipo: str = "PL",
        itens: int = 100,
        maximo: int | None = None,
        progresso=None,
    ) -> list[dict]:
        """Todas as proposicoes de um ano, seguindo o link `next`.

        A paginacao segue o link que a propria API devolve, em vez de incrementar
        um contador: assim o cliente nao precisa saber quantas paginas existem, e
        o fim e o momento em que a API para de oferecer a proxima.
        """
        relatar = progresso if progresso is not None else (lambda **_: None)
        saida: list[dict] = []
        pagina = 1
        while True:
            params = urllib.parse.urlencode(
                {
                    "ano": ano,
                    "siglaTipo": sigla_tipo,
                    "itens": itens,
                    "pagina": pagina,
                    "ordem": "ASC",
                    "ordenarPor": "id",
                }
            )
            d = self._get(f"/proposicoes?{params}")
            dados = d.get("dados", [])
            if not dados:
                break
            saida.extend(dados)
            relatar(ano=ano, pagina=pagina, total=len(saida))
            if maximo is not None and len(saida) >= maximo:
                return saida[:maximo]
            if not any(x.get("rel") == "next" for x in d.get("links", [])):
                break
            pagina += 1
            time.sleep(self.pausa_entre_chamadas)
        return saida

    def detalhe(self, id_proposicao: int) -> dict:
        return self._get(f"/proposicoes/{id_proposicao}")["dados"]

    def temas(self, id_proposicao: int) -> list[str]:
        d = self._get(f"/proposicoes/{id_proposicao}/temas")["dados"]
        return [t["tema"] for t in d if t.get("tema")]

    def coletar(
        self,
        ids: list[int],
        progresso=None,
        aviso=None,
    ) -> tuple[list[dict], list[dict]]:
        """Detalhe e temas de cada proposicao. Devolve (registros, falhas).

        Devolve as falhas em vez de engolir ou explodir: uma proposicao que nao
        responde some da amostra, e se isso acontecer em quantidade a amostra
        deixa de ser do ano e passa a ser "do que respondeu", que e outra coisa.
        """
        relatar = progresso if progresso is not None else (lambda **_: None)
        avisar = aviso if aviso is not None else (lambda _: None)
        registros: list[dict] = []
        falhas: list[dict] = []
        for i, pid in enumerate(ids, start=1):
            try:
                d = self.detalhe(pid)
                time.sleep(self.pausa_entre_chamadas)
                t = self.temas(pid)
            except FalhaDaCamara as exc:
                avisar(f"proposicao {pid}: {exc}")
                falhas.append({"id": pid, "motivo": str(exc)[:200]})
                continue
            registros.append(
                {
                    "id": pid,
                    "sigla_tipo": d.get("siglaTipo"),
                    "numero": d.get("numero"),
                    "ano": d.get("ano"),
                    "data_apresentacao": d.get("dataApresentacao"),
                    "ementa": d.get("ementa") or "",
                    "ementa_detalhada": d.get("ementaDetalhada") or "",
                    "keywords": d.get("keywords") or "",
                    "temas": t,
                }
            )
            if i % 50 == 0 or i == len(ids):
                relatar(feitos=i, total=len(ids), falhas=len(falhas))
            time.sleep(self.pausa_entre_chamadas)
        return registros, falhas
