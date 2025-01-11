# registry.py

# オペレータ名と作成関数のマッピングを保持するレジストリ
OPERATOR_REGISTRY = {}

def register_operator(name):
    """
    オペレータ作成関数をレジストリに登録するためのデコレータ

    :param name: レジストリに登録するキー（例: "activation"）
    """
    def decorator(func):
        OPERATOR_REGISTRY[name] = func
        return func
    return decorator
