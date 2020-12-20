from algorithm.method import dqn
from algorithm.method import qmix
from algorithm.method import hyq
from algorithm.method import exq
from algorithm.method import vdn

methods = dict(
	dqn=dqn.Method,
	qmix=qmix.Method,
	hyq=hyq.Method,
	exq=exq.Method,
	vdn=vdn.Method,
)