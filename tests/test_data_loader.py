import scgen

#TODO : writing test for other formats h5, csv and etc
def test_load_file():
    _data = scgen.load_file("./tests/data/train.h5ad",  backup_url="https://goo.gl/33HtVh")

