from app.utils.similarity import levenshtein, similarity_ratio


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3


def test_similarity_ratio():
    assert similarity_ratio("Adaobi Okafor", "adaobi okafor") < 1.0
    assert similarity_ratio("ada", "ada") == 1.0
    assert similarity_ratio("adaobi", "adaobu") > 0.8
