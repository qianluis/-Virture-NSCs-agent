"""PubMed 检索封装"""

import json
import os
import time
from typing import Optional


def search_pubmed(query: str, max_results: int = 10, email: str = "agent@virtualcell.ai") -> list[dict]:
    """Search PubMed via E-utilities and return structured results."""
    try:
        from Bio import Entrez
        from xml.etree import ElementTree

        Entrez.email = email

        # Search
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]

        if not ids:
            return []

        # Fetch
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
        xml_data = handle.read()
        handle.close()
        root = ElementTree.fromstring(xml_data)

        papers = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None else ""
            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None else None
            pmid_el = article.find(".//PMID")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/" if pmid_el is not None else ""
            abstract_el = article.find(".//AbstractText")
            abstract = (abstract_el.text or "")[:200] if abstract_el is not None else ""
            authors = []
            for a in article.findall(".//Author")[:5]:
                ln = a.find("LastName")
                fn = a.find("ForeName")
                if ln is not None:
                    authors.append(f"{ln.text or ''} {fn.text or ''}".strip())

            papers.append({
                "title": title,
                "source": "PubMed",
                "year": year,
                "core_contribution": abstract,
                "url": url,
                "authors": authors,
            })

        time.sleep(0.35)  # Rate limit
        return papers

    except Exception as e:
        print(f"[PubMed] Error: {e}")
        return []


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv and return structured results."""
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results,
                              sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "source": "arXiv",
                "year": result.published.year if result.published else None,
                "core_contribution": result.summary[:200],
                "url": result.entry_id,
                "authors": [a.name for a in result.authors[:5]],
            })
        return papers
    except Exception as e:
        print(f"[arXiv] Error: {e}")
        return []
