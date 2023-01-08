from myscript.helloworld import hello

def test_hello() -> None:
    message = hello()
    assert message == "Hello world!"