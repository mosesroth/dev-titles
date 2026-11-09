from dev_titles.en_terms import normalize

T = ["Android TV", "Android TV Box", "React Native", "Visual Studio Code", "OpenTelemetry"]
A = {"VS Code": "Visual Studio Code"}


def test_case_and_spacing_variants_become_the_approved_name():
    t, ch = normalize(
        "I tested on the android tv box and in visual studio code with react native.", T, A
    )
    assert t == "I tested on the Android TV Box and in Visual Studio Code with React Native."
    assert ("android tv box", "Android TV Box") in ch
    assert (
        normalize("AndroidTV apps", T, A)[0] == "Android TV apps"
    )  # only the name; 'apps' untouched
    assert normalize("open it in vs code", T, A)[0] == "open it in Visual Studio Code"


def test_code_and_correct_names_are_left_alone():
    assert normalize("npm install react-native-video", T, A)[1] == []
    assert normalize("import from @scope/react-native", T, A)[1] == []
    t, ch = normalize("Android TV and OpenTelemetry", T, A)
    assert t == "Android TV and OpenTelemetry" and ch == []


def test_lowercase_hyphenated_package_names_are_code():
    assert normalize("on npm it was react-native-video", ["React Native Video"], {})[1] == []


def test_a_longer_approved_spelling_rewraps_an_overflowing_line():
    from dev_titles.en_terms import apply
    from dev_titles.model import Cue, Subtitle

    en = Subtitle(
        "en", [Cue(0.0, 3.0, ["us a little bit about Vatoo, the products,", "and the team"])]
    )
    out, log = apply(en, ["Vantoo"], {"Vatoo": "Vantoo"})
    assert log and all(len(x) <= 42 for x in out.cues[0].lines)
    assert " ".join(out.cues[0].lines) == "us a little bit about Vantoo, the products, and the team"
