# op/base.py
import op as c
from op.registry import OPERATOR_REGISTRY
print("====== OPERATOR_REGISTRY ======\n", OPERATOR_REGISTRY)

class Operator:
    def __init__(self, op_name, network):
        """
        :param network: オペレータ名（例："conv", "activation" など）
        """
        self.op_name = op_name
        self.network = network

    def create_network(self, *args, **kwargs):
        """
        OPERATOR_REGISTRY から self.op_name に対応する関数を取得し、
        引数を渡して実行する
        """
        func = OPERATOR_REGISTRY.get(self.op_name)
        print("OPERATOR_REGISTRY.get(self.op_name): ", func)

        if callable(func):
            print(f"Calling function for operator '{self.op_name}' with args: {args}, kwargs: {kwargs}")
            return func(self.network, *args, **kwargs)
        else:
            raise AttributeError(f"Operator '{self.op_name}' はレジストリに登録されていません。")
