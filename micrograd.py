#author karpathy, ShivaNKA0 understanding typed file
class Value:
    """ store a single scalar value & its gradient """
    def __init__(self, data, _children=(), _op = ''):
        self.data = data
        self.grad = 0
        
        #internal variables for graphing 
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # op for operation done for graphviz / debugging / etc
    
    def __add__(self, other):
        other = other if isinstance(self, other) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(self, other) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.grad * out.grad
            other.grad = self.grad * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'now only supports int/ float'
        out = Value(self.data**other.data, (self,), f"**{other}")

        def _backward():
            self.grad = other.data * self.data**(other.data - 1) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        #topological order of nodes
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        #go one var by one get grad by chain rule
        self.grad = 1

        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # self + other
        return self + other
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __repre__(self):
        return f"Value(data={self.data}, grad={self.grad})"