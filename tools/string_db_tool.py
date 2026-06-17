"""STRING DB / KEGG API 封装"""

import requests


def search_string_db(gene: str, species: str = "9606") -> dict:
    """
    查询 STRING DB 获取蛋白质互作网络。

    Args:
        gene: 基因名（如 NOTCH1）
        species: NCBI taxonomy ID (9606 = human)

    Returns:
        包含互作伙伴列表的字典
    """
    url = "https://string-db.org/api/json/interaction_partners"
    params = {
        "identifiers": gene,
        "species": species,
        "limit": 20,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {
            "gene": gene,
            "interactions": [
                {
                    "partner": item.get("preferredName_B", ""),
                    "score": item.get("score", 0),
                }
                for item in data
            ],
        }
    except Exception as e:
        return {"gene": gene, "error": str(e), "interactions": []}


def get_kegg_pathway(pathway_id: str) -> dict:
    """
    查询 KEGG 通路详情。

    Args:
        pathway_id: KEGG 通路 ID (如 ko04330 = Notch)

    Returns:
        通路基本信息
    """
    url = f"https://rest.kegg.jp/get/{pathway_id}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        text = r.text
        result = {"id": pathway_id, "raw": text[:2000]}
        # Extract name
        for line in text.split("\n"):
            if line.startswith("NAME"):
                result["name"] = line[4:].strip()
                break
        return result
    except Exception as e:
        return {"id": pathway_id, "error": str(e)}
