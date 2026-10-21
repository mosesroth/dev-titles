"""Names policy: partner companies stay Latin; ordinary-word products get one rendering per video."""

import json


def test_apply_policy_keeps_companies_latin_and_unifies_product_renderings():
    from dev_titles.entities import apply_policy

    vv = {"names": ["Sam"], "renderings": {
        "zh-Hans": {"Acme Air": {"canonical": "阿克米航空", "avoid": ["Acme Air"]},
                    "concierge": {"canonical": "礼宾服务"}, "Concierge": {"canonical": "Concierge", "avoid": ["礼宾服务"]}},
        "ja": {"Acme Air": {"canonical": "アクメ航空"}}}}
    pol = {"companies": ["Acme Air"], "ordinary_products": [{"name": "Concierge", "ja": "コンシェルジュ", "zh-Hans": "礼宾服务"}]}
    out = apply_policy(vv, pol)
    assert "Acme Air" in out["names"] and "Sam" in out["names"]
    assert "Acme Air" not in out["renderings"]["zh-Hans"] and "Acme Air" not in out["renderings"]["ja"]
    zh = out["renderings"]["zh-Hans"]
    assert list(k for k in zh if k.lower() == "concierge") == ["Concierge"] and zh["Concierge"]["canonical"] == "礼宾服务"
    assert out["renderings"]["ja"]["Concierge"]["canonical"] == "コンシェルジュ"
    assert vv["renderings"]["zh-Hans"]["concierge"]  # input not mutated


def test_names_policy_filters_amazon_brands_and_caches(tmp_path):
    from dev_titles.entities import names_policy

    calls = []

    def inv(p):
        calls.append(p)
        return json.dumps({"companies": ["Acme Air"],
                           "ordinary_products": [
                                                 {"name": "Concierge", "ja": "コンシェルジュ", "zh-Hans": "礼宾服务"}]})

    cp = tmp_path / "entities.json"
    pol = names_policy("Acme Air built Concierge. Acme Air flies.", inv, cp, brief="b")
    assert pol["companies"] == ["Acme Air"] and [x["name"] for x in pol["ordinary_products"]] == ["Concierge"]
    assert names_policy("Acme Air built Concierge. Acme Air flies.", inv, cp, brief="b") == pol and len(calls) == 1
