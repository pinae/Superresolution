# -*- coding: utf-8 -*-
from tensorflow.compat.v1 import placeholder, Session, disable_eager_execution
from tensorflow import float32
disable_eager_execution()

a = placeholder(float32)
b = placeholder(float32)
c = placeholder(float32)
x = (a + b) / c
sess = Session()
result = sess.run([x], {
    a: 2, b: 3, c: 4
})
print(result)  # [1.25]
