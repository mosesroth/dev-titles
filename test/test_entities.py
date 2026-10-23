"""Names policy: partner companies stay Latin; ordinary-word products get one rendering per video."""

import json


def test_apply_policy_keeps_companies_latin_and_unifies_product_renderings():
    from dev_titles.entities import apply_policy

    vv = {
        "names": ["Sam"],
        "renderings": {
            "zh-Hans": {
                "Acme Air": {"canonical": "阿克米航空", "avoid": ["Acme Air"]},
                "concierge": {"canonical": "礼宾服务"},
                "Concierge": {"canonical": "Concierge", "avoid": ["礼宾服务"]},
            },
            "ja": {"Acme Air": {"canonical": "アクメ航空"}},
        },
    }
    pol = {
        "companies": ["Acme Air"],
        "company_names": {"Acme Air": {"zh-Hans": "阿克米航空"}},
        "ordinary_products": [{"name": "Concierge", "ja": "コンシェルジュ", "zh-Hans": "礼宾服务"}],
    }
    out = apply_policy(vv, pol)
    assert out["names"] == ["Sam"]
    assert (
        out["renderings"]["zh-Hans"]["Acme Air"]["canonical"] == "阿克米航空"
    )  # partner's official zh name
    assert out["renderings"]["ja"]["Acme Air"] == {"canonical": "Acme Air"}  # none in ja -> Latin
    zh = out["renderings"]["zh-Hans"]
    assert (
        list(k for k in zh if k.lower() == "concierge") == ["Concierge"]
        and zh["Concierge"]["canonical"] == "礼宾服务"
    )
    assert out["renderings"]["ja"]["Concierge"]["canonical"] == "コンシェルジュ"
    assert vv["renderings"]["zh-Hans"]["concierge"]  # input not mutated


def test_names_policy_filters_amazon_brands_and_caches(tmp_path):
    from dev_titles.entities import names_policy

    calls = []

    def inv(p):
        calls.append(p)
        return json.dumps(
            {
                "companies": ["Acme Air"],
                "ordinary_products": [
                    {"name": "Concierge", "ja": "コンシェルジュ", "zh-Hans": "礼宾服务"}
                ],
            }
        )

    cp = tmp_path / "entities.json"
    pol = names_policy("Acme Air built Concierge. Acme Air flies.", inv, cp, brief="b")
    assert pol["companies"] == ["Acme Air"] and [x["name"] for x in pol["ordinary_products"]] == [
        "Concierge"
    ]
    assert (
        names_policy("Acme Air built Concierge. Acme Air flies.", inv, cp, brief="b") == pol
        and len(calls) == 1
    )


def test_official_rendering_only_from_the_partners_own_page_and_locked(tmp_path):
    from dev_titles.entities import load_partners, official_rendering, resolve_official

    pages = {
        "https://www.acmeair.com/zh-cn/ai": "欢迎使用 Acme Air 的礼宾服务 随时为您规划旅程 阿克米航空",
        "https://blog.example.com/acme": "Acme Air 推出了礼宾助手",
    }
    search = lambda q: list(pages)
    fetch = lambda u: pages[u]
    judge = lambda p: (
        '{"rendering": "阿克米航空", "quote": "随时为您规划旅程 阿克米航空"}'
        if "OWN name" in p
        else (
            '{"rendering": "礼宾服务", "quote": "欢迎使用 Acme Air 的礼宾服务"}'
            if "acmeair.com" in p
            else '{"rendering": "礼宾助手", "quote": "推出了礼宾助手"}'
        )
    )
    assert official_rendering(
        "Acme Air", "Concierge", "zh-Hans", web_search=search, web_fetch=fetch, invoke=judge
    ) == ("礼宾服务", "https://www.acmeair.com/zh-cn/ai")
    fake = (
        lambda p: '{"rendering": "礼宾中心", "quote": "不在页面上的句子礼宾中心"}'
    )  # quote not on the page
    assert official_rendering(
        "Acme Air", "Concierge", "zh-Hans", web_search=search, web_fetch=fetch, invoke=fake
    ) == (None, None)
    path = tmp_path / "partners.yaml"
    pol = {
        "companies": ["Acme Air"],
        "ordinary_products": [
            {"name": "Concierge", "company": "Acme Air", "ja": "コンシェルジュ", "zh-Hans": "礼宾"}
        ],
    }
    out = resolve_official(
        pol,
        web_search=search,
        web_fetch=fetch,
        invoke=judge,
        langs=("zh-Hans",),
        path=path,
        logger=lambda m: None,
    )
    assert out["ordinary_products"][0]["zh-Hans"] == "礼宾服务"
    assert out["company_names"]["Acme Air"]["zh-Hans"] == "阿克米航空"
    assert load_partners(path)["Acme Air"]["name"]["zh-Hans"]["rendering"] == "阿克米航空"
    assert (
        load_partners(path)["Acme Air"]["products"]["Concierge"]["zh-Hans"]["rendering"]
        == "礼宾服务"
    )
    # locked: a second run never searches again
    boom = lambda q: (_ for _ in ()).throw(AssertionError("searched"))
    pol2 = {
        "companies": ["Acme Air"],
        "ordinary_products": [{"name": "Concierge", "company": "Acme Air", "zh-Hans": "x"}],
    }
    assert (
        resolve_official(
            pol2,
            web_search=boom,
            web_fetch=fetch,
            invoke=judge,
            langs=("zh-Hans",),
            path=path,
            logger=lambda m: None,
        )["ordinary_products"][0]["zh-Hans"]
        == "礼宾服务"
    )


def test_company_local_name_only_when_the_site_brands_itself_that_way():
    from dev_titles.entities import official_company_name

    page = {
        "https://www.acmepay.com/jp/": "Acme Payの使い方 Acme Payで支払う Acme Pay（アクメペイ）とは"
    }
    judge = lambda p: '{"rendering": "アクメペイ", "quote": "Acme Pay（アクメペイ）とは"}'
    assert official_company_name(
        "Acme Pay", "ja", web_search=lambda q: list(page), web_fetch=page.get, invoke=judge
    ) == (
        None,
        None,
    )  # Latin 3x vs local 1x -> stays Latin
