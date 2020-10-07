class CollectiveOperator:
	def __init__(self, local_op, collective, mpi_op = 'sum'):
		assert hasattr(local_op,'mult')
		self.local_op = local_op
		self.collective = collective
		self.mpi_op = mpi_op

	def mult(self, x,y):
		self.local_op.mult(x,y)
		self.collective.allReduce(y, self.mpi_op)

	def transpmult(self,x,y):
		assert hasattr(self.local_op, 'transpmult')
		self.local_op.transpmult(x,y)
		self.collective.allReduce(y,self.mpi_op)

	def init_vector(self,x,dim):
		self.local_op.init_vector(x,dim)

